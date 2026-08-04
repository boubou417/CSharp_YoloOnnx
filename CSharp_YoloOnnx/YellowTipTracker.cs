using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace CSharp_YoloOnnx
{
    /// <summary>
    /// Tracks a saturated yellow marker. Once acquired, a velocity-predicted
    /// region is searched at full pixel resolution; full-frame sampling is
    /// used only for initial acquisition and fallback.
    /// </summary>
    public sealed class YellowTipTracker
    {
        private const int FullFrameSampleStep = 3;
        private const int RoiSampleStep = 1;
        private const int MinimumFullFrameSamples = 6;
        private const int MinimumRoiSamples = 18;
        private const int PredictionMemoryMs = 500;
        private const float MinimumRoiRadius = 140f;
        private const float MaximumRoiRadius = 420f;
        private const float RoiMotionExpansion = 1.8f;
        private const float PreviousPositionWeight = 0.02f;

        private bool[] mask;
        private int[] queue;
        private PointF? previousPoint;
        private PointF velocityPixelsPerMillisecond;
        private DateTime previousPointSeenAt = DateTime.MinValue;

        public void Reset()
        {
            previousPoint = null;
            velocityPixelsPerMillisecond = PointF.Empty;
            previousPointSeenAt = DateTime.MinValue;
        }

        public unsafe bool TryDetect(
            Bitmap frame,
            out PointF tip,
            out RectangleF bounds)
        {
            tip = PointF.Empty;
            bounds = RectangleF.Empty;

            if (frame == null ||
                frame.Width < FullFrameSampleStep ||
                frame.Height < FullFrameSampleStep)
            {
                return false;
            }

            Rectangle frameBounds =
                new Rectangle(0, 0, frame.Width, frame.Height);
            BitmapData data = frame.LockBits(
                frameBounds,
                ImageLockMode.ReadOnly,
                frame.PixelFormat);

            try
            {
                int bytesPerPixel =
                    Image.GetPixelFormatSize(
                        frame.PixelFormat) / 8;

                if (bytesPerPixel < 3)
                    return false;

                DateTime now = DateTime.UtcNow;
                PointF prediction = PointF.Empty;
                bool hasPrediction =
                    previousPoint.HasValue &&
                    previousPointSeenAt != DateTime.MinValue &&
                    (now - previousPointSeenAt)
                        .TotalMilliseconds <=
                    PredictionMemoryMs;

                if (hasPrediction)
                {
                    double elapsedMilliseconds =
                        Math.Max(
                            1d,
                            (now - previousPointSeenAt)
                                .TotalMilliseconds);
                    prediction = new PointF(
                        previousPoint.Value.X +
                        velocityPixelsPerMillisecond.X *
                        (float)elapsedMilliseconds,
                        previousPoint.Value.Y +
                        velocityPixelsPerMillisecond.Y *
                        (float)elapsedMilliseconds);
                    float speed =
                        (float)Math.Sqrt(
                            velocityPixelsPerMillisecond.X *
                            velocityPixelsPerMillisecond.X +
                            velocityPixelsPerMillisecond.Y *
                            velocityPixelsPerMillisecond.Y);
                    float radius =
                        Math.Max(
                            MinimumRoiRadius,
                            Math.Min(
                                MaximumRoiRadius,
                                MinimumRoiRadius +
                                speed *
                                (float)elapsedMilliseconds *
                                RoiMotionExpansion));
                    Rectangle roi =
                        IntersectWithFrame(
                            prediction,
                            radius,
                            frame.Size);

                    if (TryFindYellowComponent(
                        data,
                        frame.Size,
                        bytesPerPixel,
                        roi,
                        RoiSampleStep,
                        MinimumRoiSamples,
                        prediction,
                        out tip,
                        out bounds))
                    {
                        UpdateMotion(tip, now);
                        return true;
                    }
                }

                PointF reference =
                    previousPoint ?? PointF.Empty;

                if (TryFindYellowComponent(
                    data,
                    frame.Size,
                    bytesPerPixel,
                    frameBounds,
                    FullFrameSampleStep,
                    MinimumFullFrameSamples,
                    reference,
                    out tip,
                    out bounds))
                {
                    UpdateMotion(tip, now);
                    return true;
                }

                return false;
            }
            finally
            {
                frame.UnlockBits(data);
            }
        }

        private unsafe bool TryFindYellowComponent(
            BitmapData data,
            Size frameSize,
            int bytesPerPixel,
            Rectangle searchRegion,
            int sampleStep,
            int minimumSamples,
            PointF referencePoint,
            out PointF tip,
            out RectangleF bounds)
        {
            tip = PointF.Empty;
            bounds = RectangleF.Empty;

            if (searchRegion.Width <= 0 ||
                searchRegion.Height <= 0)
            {
                return false;
            }

            int width =
                (searchRegion.Width +
                 sampleStep - 1) /
                sampleStep;
            int height =
                (searchRegion.Height +
                 sampleStep - 1) /
                sampleStep;
            int requiredLength = width * height;
            EnsureBuffers(requiredLength);
            Array.Clear(mask, 0, requiredLength);
            byte* scan0 = (byte*)data.Scan0.ToPointer();

            for (int gy = 0; gy < height; gy++)
            {
                int sourceY = Math.Min(
                    searchRegion.Bottom - 1,
                    searchRegion.Top +
                    gy * sampleStep);
                byte* row =
                    data.Stride >= 0
                        ? scan0 + sourceY * data.Stride
                        : scan0 +
                          (frameSize.Height -
                           1 -
                           sourceY) *
                          -data.Stride;

                for (int gx = 0; gx < width; gx++)
                {
                    int sourceX = Math.Min(
                        searchRegion.Right - 1,
                        searchRegion.Left +
                        gx * sampleStep);
                    byte* pixel =
                        row + sourceX * bytesPerPixel;

                    mask[gy * width + gx] =
                        IsYellow(
                            pixel[2],
                            pixel[1],
                            pixel[0]);
                }
            }

            int bestCount = 0;
            float bestScore = float.MinValue;
            float bestSumX = 0f;
            float bestSumY = 0f;
            int bestMinX = 0;
            int bestMinY = 0;
            int bestMaxX = 0;
            int bestMaxY = 0;

            for (int index = 0;
                index < requiredLength;
                index++)
            {
                if (!mask[index])
                    continue;

                int head = 0;
                int tail = 0;
                queue[tail++] = index;
                mask[index] = false;
                int count = 0;
                float sumX = 0f;
                float sumY = 0f;
                int minX = width;
                int minY = height;
                int maxX = 0;
                int maxY = 0;

                while (head < tail)
                {
                    int current = queue[head++];
                    int x = current % width;
                    int y = current / width;
                    count++;
                    sumX += x;
                    sumY += y;
                    minX = Math.Min(minX, x);
                    minY = Math.Min(minY, y);
                    maxX = Math.Max(maxX, x);
                    maxY = Math.Max(maxY, y);

                    // A two-sample neighbourhood joins thin or slightly
                    // broken streaks created by fast marker motion.
                    for (int offsetY = -2;
                        offsetY <= 2;
                        offsetY++)
                    {
                        for (int offsetX = -2;
                            offsetX <= 2;
                            offsetX++)
                        {
                            if (offsetX == 0 &&
                                offsetY == 0)
                            {
                                continue;
                            }

                            EnqueueIfYellow(
                                x + offsetX,
                                y + offsetY,
                                width,
                                height,
                                ref tail);
                        }
                    }
                }

                if (count < minimumSamples)
                    continue;

                float centerX =
                    searchRegion.Left +
                    sumX / count *
                    sampleStep;
                float centerY =
                    searchRegion.Top +
                    sumY / count *
                    sampleStep;
                float dx =
                    centerX - referencePoint.X;
                float dy =
                    centerY - referencePoint.Y;
                float distance =
                    (float)Math.Sqrt(dx * dx + dy * dy);
                float score =
                    count -
                    distance *
                    PreviousPositionWeight;

                if (score <= bestScore)
                    continue;

                bestScore = score;
                bestCount = count;
                bestSumX = sumX;
                bestSumY = sumY;
                bestMinX = minX;
                bestMinY = minY;
                bestMaxX = maxX;
                bestMaxY = maxY;
            }

            if (bestCount < minimumSamples)
                return false;

            tip = new PointF(
                searchRegion.Left +
                bestSumX / bestCount *
                sampleStep,
                searchRegion.Top +
                bestSumY / bestCount *
                sampleStep);
            bounds = RectangleF.FromLTRB(
                searchRegion.Left +
                bestMinX * sampleStep,
                searchRegion.Top +
                bestMinY * sampleStep,
                Math.Min(
                    frameSize.Width,
                    searchRegion.Left +
                    (bestMaxX + 1) *
                    sampleStep),
                Math.Min(
                    frameSize.Height,
                    searchRegion.Top +
                    (bestMaxY + 1) *
                    sampleStep));
            return true;
        }

        private void UpdateMotion(
            PointF point,
            DateTime now)
        {
            if (previousPoint.HasValue &&
                previousPointSeenAt != DateTime.MinValue)
            {
                double elapsedMilliseconds =
                    Math.Max(
                        1d,
                        (now - previousPointSeenAt)
                            .TotalMilliseconds);

                if (elapsedMilliseconds <=
                    PredictionMemoryMs)
                {
                    float instantVelocityX =
                        (point.X -
                         previousPoint.Value.X) /
                        (float)elapsedMilliseconds;
                    float instantVelocityY =
                        (point.Y -
                         previousPoint.Value.Y) /
                        (float)elapsedMilliseconds;
                    velocityPixelsPerMillisecond =
                        new PointF(
                            velocityPixelsPerMillisecond.X *
                            0.55f +
                            instantVelocityX *
                            0.45f,
                            velocityPixelsPerMillisecond.Y *
                            0.55f +
                            instantVelocityY *
                            0.45f);
                }
                else
                {
                    velocityPixelsPerMillisecond =
                        PointF.Empty;
                }
            }

            previousPoint = point;
            previousPointSeenAt = now;
        }

        private static Rectangle IntersectWithFrame(
            PointF center,
            float radius,
            Size frameSize)
        {
            int left =
                Math.Max(
                    0,
                    (int)Math.Floor(
                        center.X - radius));
            int top =
                Math.Max(
                    0,
                    (int)Math.Floor(
                        center.Y - radius));
            int right =
                Math.Min(
                    frameSize.Width,
                    (int)Math.Ceiling(
                        center.X + radius));
            int bottom =
                Math.Min(
                    frameSize.Height,
                    (int)Math.Ceiling(
                        center.Y + radius));

            if (right <= left ||
                bottom <= top)
            {
                return Rectangle.Empty;
            }

            return Rectangle.FromLTRB(
                left,
                top,
                right,
                bottom);
        }

        private void EnsureBuffers(int requiredLength)
        {
            if (mask == null ||
                mask.Length < requiredLength)
            {
                mask = new bool[requiredLength];
                queue = new int[requiredLength];
            }
        }

        private void EnqueueIfYellow(
            int x,
            int y,
            int width,
            int height,
            ref int tail)
        {
            if (x < 0 || y < 0 ||
                x >= width || y >= height)
            {
                return;
            }

            int index = y * width + x;

            if (!mask[index])
                return;

            mask[index] = false;
            queue[tail++] = index;
        }

        private static bool IsYellow(
            byte red,
            byte green,
            byte blue)
        {
            float r = red / 255f;
            float g = green / 255f;
            float b = blue / 255f;
            float maximum =
                Math.Max(r, Math.Max(g, b));
            float minimum =
                Math.Min(r, Math.Min(g, b));
            float chroma = maximum - minimum;

            if (maximum < 0.52f ||
                chroma < 0.16f)
            {
                return false;
            }

            float saturation =
                maximum <= 0f
                    ? 0f
                    : chroma / maximum;

            if (saturation < 0.31f)
                return false;

            float hue;

            if (maximum == r)
            {
                hue =
                    60f *
                    (((g - b) / chroma) % 6f);
            }
            else if (maximum == g)
            {
                hue =
                    60f *
                    (((b - r) / chroma) + 2f);
            }
            else
            {
                hue =
                    60f *
                    (((r - g) / chroma) + 4f);
            }

            if (hue < 0f)
                hue += 360f;

            return hue >= 36f &&
                hue <= 78f;
        }
    }
}
