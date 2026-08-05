using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace CSharp_YoloOnnx
{
    /// <summary>
    /// Finds a thin red tube near either pose wrist and returns the red
    /// component point closest to that wrist. The wrist association keeps
    /// unrelated red objects in a wide-angle background from becoming the
    /// drawing point.
    /// </summary>
    public sealed class RedTubeTracker
    {
        private const double ReferencePixelCount = 1600000d;
        private const double HighResolutionPixelCount = 3000000d;
        private const float SearchRadius = 220f;
        private const float MaximumWristDistance = 135f;
        private const int MinimumFullResolutionSamples = 32;
        private const int MinimumHighResolutionSamples = 12;
        private const float MinimumTubeSpan = 22f;
        private const float MaximumTubeFillRatio = 0.62f;
        private const float MinimumTubeElongation = 1.60f;
        private const int PreviousPointMemoryMs = 300;
        private const float PreviousPointWeight = 0.18f;

        private bool[] mask;
        private int[] queue;
        private PointF? previousPoint;
        private DateTime previousPointSeenAt = DateTime.MinValue;

        private sealed class Candidate
        {
            public PointF Point;
            public RectangleF Bounds;
            public float Score = float.MaxValue;
        }

        public void Reset()
        {
            previousPoint = null;
            previousPointSeenAt = DateTime.MinValue;
        }

        public unsafe bool TryDetect(
            Bitmap frame,
            PointF? leftWrist,
            PointF? rightWrist,
            out PointF drawingPoint,
            out RectangleF bounds)
        {
            drawingPoint = PointF.Empty;
            bounds = RectangleF.Empty;

            if (frame == null ||
                frame.Width < 3 ||
                frame.Height < 3 ||
                (!leftWrist.HasValue &&
                 !rightWrist.HasValue))
            {
                ResetExpiredPoint(DateTime.UtcNow);
                return false;
            }

            Rectangle frameBounds =
                new Rectangle(
                    0,
                    0,
                    frame.Width,
                    frame.Height);
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
                double pixelCount =
                    (double)frame.Width *
                    frame.Height;
                float resolutionScale =
                    (float)Math.Sqrt(
                        Math.Max(
                            1d,
                            pixelCount) /
                        ReferencePixelCount);
                resolutionScale =
                    Math.Max(
                        0.80f,
                        Math.Min(1.40f, resolutionScale));
                int sampleStep =
                    pixelCount >=
                        HighResolutionPixelCount
                        ? 2
                        : 1;
                int minimumSamples =
                    sampleStep == 1
                        ? MinimumFullResolutionSamples
                        : MinimumHighResolutionSamples;
                Candidate best = new Candidate();

                if (leftWrist.HasValue &&
                    IsInsideFrame(
                        leftWrist.Value,
                        frame.Size))
                {
                    FindBestNearWrist(
                        data,
                        frame.Size,
                        bytesPerPixel,
                        leftWrist.Value,
                        sampleStep,
                        minimumSamples,
                        resolutionScale,
                        now,
                        best);
                }

                if (rightWrist.HasValue &&
                    IsInsideFrame(
                        rightWrist.Value,
                        frame.Size))
                {
                    FindBestNearWrist(
                        data,
                        frame.Size,
                        bytesPerPixel,
                        rightWrist.Value,
                        sampleStep,
                        minimumSamples,
                        resolutionScale,
                        now,
                        best);
                }

                if (best.Score == float.MaxValue)
                {
                    ResetExpiredPoint(now);
                    return false;
                }

                drawingPoint = best.Point;
                bounds = best.Bounds;
                previousPoint = drawingPoint;
                previousPointSeenAt = now;
                return true;
            }
            finally
            {
                frame.UnlockBits(data);
            }
        }

        private unsafe void FindBestNearWrist(
            BitmapData data,
            Size frameSize,
            int bytesPerPixel,
            PointF wrist,
            int sampleStep,
            int minimumSamples,
            float resolutionScale,
            DateTime now,
            Candidate best)
        {
            float searchRadius =
                SearchRadius * resolutionScale;
            Rectangle region =
                IntersectWithFrame(
                    wrist,
                    searchRadius,
                    frameSize);

            if (region.Width <= 0 ||
                region.Height <= 0)
            {
                return;
            }

            int width =
                (region.Width +
                 sampleStep - 1) /
                sampleStep;
            int height =
                (region.Height +
                 sampleStep - 1) /
                sampleStep;
            int requiredLength = width * height;
            EnsureBuffers(requiredLength);
            Array.Clear(mask, 0, requiredLength);
            byte* scan0 =
                (byte*)data.Scan0.ToPointer();

            for (int gridY = 0;
                gridY < height;
                gridY++)
            {
                int sourceY = Math.Min(
                    region.Bottom - 1,
                    region.Top +
                    gridY * sampleStep);
                byte* row =
                    data.Stride >= 0
                        ? scan0 +
                          sourceY * data.Stride
                        : scan0 +
                          (frameSize.Height -
                           1 - sourceY) *
                          -data.Stride;

                for (int gridX = 0;
                    gridX < width;
                    gridX++)
                {
                    int sourceX = Math.Min(
                        region.Right - 1,
                        region.Left +
                        gridX * sampleStep);
                    byte* pixel =
                        row +
                        sourceX * bytesPerPixel;
                    mask[gridY * width + gridX] =
                        IsRedTube(
                            pixel[2],
                            pixel[1],
                            pixel[0]);
                }
            }

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
                int minX = width;
                int minY = height;
                int maxX = 0;
                int maxY = 0;
                double sumX = 0d;
                double sumY = 0d;
                double sumXX = 0d;
                double sumYY = 0d;
                double sumXY = 0d;
                float nearestSquared =
                    float.MaxValue;
                PointF nearestPoint =
                    PointF.Empty;

                while (head < tail)
                {
                    int current = queue[head++];
                    int gridX = current % width;
                    int gridY = current / width;
                    int sourceX =
                        region.Left +
                        gridX * sampleStep;
                    int sourceY =
                        region.Top +
                        gridY * sampleStep;
                    count++;
                    minX = Math.Min(minX, gridX);
                    minY = Math.Min(minY, gridY);
                    maxX = Math.Max(maxX, gridX);
                    maxY = Math.Max(maxY, gridY);
                    sumX += sourceX;
                    sumY += sourceY;
                    sumXX += sourceX * sourceX;
                    sumYY += sourceY * sourceY;
                    sumXY += sourceX * sourceY;
                    float wristDx =
                        sourceX - wrist.X;
                    float wristDy =
                        sourceY - wrist.Y;
                    float wristDistanceSquared =
                        wristDx * wristDx +
                        wristDy * wristDy;

                    if (wristDistanceSquared <
                        nearestSquared)
                    {
                        nearestSquared =
                            wristDistanceSquared;
                        nearestPoint =
                            new PointF(
                                sourceX,
                                sourceY);
                    }

                    for (int offsetY = -1;
                        offsetY <= 1;
                        offsetY++)
                    {
                        for (int offsetX = -1;
                            offsetX <= 1;
                            offsetX++)
                        {
                            if (offsetX == 0 &&
                                offsetY == 0)
                            {
                                continue;
                            }

                            EnqueueIfRed(
                                gridX + offsetX,
                                gridY + offsetY,
                                width,
                                height,
                                ref tail);
                        }
                    }
                }

                if (count < minimumSamples)
                    continue;

                int componentWidth =
                    maxX - minX + 1;
                int componentHeight =
                    maxY - minY + 1;
                float span =
                    Math.Max(
                        componentWidth,
                        componentHeight) *
                    sampleStep;

                if (span <
                    MinimumTubeSpan *
                    resolutionScale)
                {
                    continue;
                }

                float fillRatio =
                    count /
                    (float)(componentWidth *
                            componentHeight);

                if (fillRatio >
                    MaximumTubeFillRatio)
                {
                    continue;
                }

                float nearestDistance =
                    (float)Math.Sqrt(
                        nearestSquared);

                if (nearestDistance >
                    MaximumWristDistance *
                    resolutionScale)
                {
                    continue;
                }

                double meanX = sumX / count;
                double meanY = sumY / count;
                double covarianceXX =
                    sumXX / count -
                    meanX * meanX;
                double covarianceYY =
                    sumYY / count -
                    meanY * meanY;
                double covarianceXY =
                    sumXY / count -
                    meanX * meanY;
                double trace =
                    covarianceXX +
                    covarianceYY;
                double determinant =
                    covarianceXX *
                    covarianceYY -
                    covarianceXY *
                    covarianceXY;
                double discriminant =
                    Math.Sqrt(
                        Math.Max(
                            0d,
                            trace * trace * 0.25d -
                            determinant));
                double major =
                    trace * 0.5d +
                    discriminant;
                double minor =
                    Math.Max(
                        0.25d,
                        trace * 0.5d -
                        discriminant);

                if (major / minor <
                    MinimumTubeElongation)
                {
                    continue;
                }

                float temporalPenalty = 0f;

                if (previousPoint.HasValue &&
                    previousPointSeenAt !=
                        DateTime.MinValue &&
                    (now - previousPointSeenAt)
                        .TotalMilliseconds <=
                    PreviousPointMemoryMs)
                {
                    float previousDx =
                        nearestPoint.X -
                        previousPoint.Value.X;
                    float previousDy =
                        nearestPoint.Y -
                        previousPoint.Value.Y;
                    temporalPenalty =
                        (float)Math.Sqrt(
                            previousDx * previousDx +
                            previousDy * previousDy) *
                        PreviousPointWeight;
                }

                float score =
                    nearestDistance +
                    temporalPenalty -
                    Math.Min(160f, span) *
                    0.03f;

                if (score >= best.Score)
                    continue;

                best.Score = score;
                best.Point = nearestPoint;
                best.Bounds =
                    RectangleF.FromLTRB(
                        region.Left +
                        minX * sampleStep,
                        region.Top +
                        minY * sampleStep,
                        Math.Min(
                            frameSize.Width,
                            region.Left +
                            (maxX + 1) *
                            sampleStep),
                        Math.Min(
                            frameSize.Height,
                            region.Top +
                            (maxY + 1) *
                            sampleStep));
            }
        }

        private void ResetExpiredPoint(
            DateTime now)
        {
            if (previousPointSeenAt ==
                    DateTime.MinValue ||
                (now - previousPointSeenAt)
                    .TotalMilliseconds <=
                PreviousPointMemoryMs)
            {
                return;
            }

            Reset();
        }

        private static bool IsInsideFrame(
            PointF point,
            Size frameSize)
        {
            return point.X >= 0f &&
                point.Y >= 0f &&
                point.X < frameSize.Width &&
                point.Y < frameSize.Height;
        }

        private static Rectangle IntersectWithFrame(
            PointF center,
            float radius,
            Size frameSize)
        {
            int left = Math.Max(
                0,
                (int)Math.Floor(
                    center.X - radius));
            int top = Math.Max(
                0,
                (int)Math.Floor(
                    center.Y - radius));
            int right = Math.Min(
                frameSize.Width,
                (int)Math.Ceiling(
                    center.X + radius));
            int bottom = Math.Min(
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

        private void EnsureBuffers(
            int requiredLength)
        {
            if (mask != null &&
                mask.Length >= requiredLength)
            {
                return;
            }

            mask = new bool[requiredLength];
            queue = new int[requiredLength];
        }

        private void EnqueueIfRed(
            int x,
            int y,
            int width,
            int height,
            ref int tail)
        {
            if (x < 0 ||
                y < 0 ||
                x >= width ||
                y >= height)
            {
                return;
            }

            int index = y * width + x;

            if (!mask[index])
                return;

            mask[index] = false;
            queue[tail++] = index;
        }

        private static bool IsRedTube(
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

            if (maximum < 0.30f ||
                chroma < 0.16f)
            {
                return false;
            }

            float saturation =
                maximum <= 0f
                    ? 0f
                    : chroma / maximum;

            if (saturation < 0.52f)
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

            return hue <= 24f ||
                hue >= 348f;
        }
    }
}
