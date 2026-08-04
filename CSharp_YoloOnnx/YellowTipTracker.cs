using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace CSharp_YoloOnnx
{
    /// <summary>
    /// Tracks the largest connected high-saturation yellow marker directly
    /// from the camera frame. Sampling every third pixel keeps the tracker
    /// lightweight enough to run on the acquisition thread.
    /// </summary>
    public sealed class YellowTipTracker
    {
        private const int SampleStep = 3;
        private const int MinimumComponentSamples = 10;
        private const float PreviousPositionWeight = 0.12f;

        private bool[] mask;
        private int[] queue;
        private int gridWidth;
        private int gridHeight;
        private PointF? previousPoint;

        public void Reset()
        {
            previousPoint = null;
        }

        public unsafe bool TryDetect(
            Bitmap frame,
            out PointF tip,
            out RectangleF bounds)
        {
            tip = PointF.Empty;
            bounds = RectangleF.Empty;

            if (frame == null ||
                frame.Width < SampleStep ||
                frame.Height < SampleStep)
            {
                return false;
            }

            int width = (frame.Width + SampleStep - 1) / SampleStep;
            int height = (frame.Height + SampleStep - 1) / SampleStep;
            EnsureBuffers(width, height);
            Array.Clear(mask, 0, width * height);

            Rectangle frameBounds =
                new Rectangle(0, 0, frame.Width, frame.Height);
            BitmapData data = frame.LockBits(
                frameBounds,
                ImageLockMode.ReadOnly,
                frame.PixelFormat);

            try
            {
                int bytesPerPixel =
                    Image.GetPixelFormatSize(frame.PixelFormat) / 8;

                if (bytesPerPixel < 3)
                    return false;

                byte* scan0 = (byte*)data.Scan0.ToPointer();

                for (int gy = 0; gy < height; gy++)
                {
                    int sourceY = Math.Min(
                        frame.Height - 1,
                        gy * SampleStep);
                    byte* row =
                        data.Stride >= 0
                            ? scan0 + sourceY * data.Stride
                            : scan0 +
                              (frame.Height - 1 - sourceY) *
                              -data.Stride;

                    for (int gx = 0; gx < width; gx++)
                    {
                        int sourceX = Math.Min(
                            frame.Width - 1,
                            gx * SampleStep);
                        byte* pixel =
                            row + sourceX * bytesPerPixel;

                        mask[gy * width + gx] =
                            IsYellow(
                                pixel[2],
                                pixel[1],
                                pixel[0]);
                    }
                }
            }
            finally
            {
                frame.UnlockBits(data);
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
                index < width * height;
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

                    EnqueueIfYellow(
                        x - 1,
                        y,
                        width,
                        height,
                        ref tail);
                    EnqueueIfYellow(
                        x + 1,
                        y,
                        width,
                        height,
                        ref tail);
                    EnqueueIfYellow(
                        x,
                        y - 1,
                        width,
                        height,
                        ref tail);
                    EnqueueIfYellow(
                        x,
                        y + 1,
                        width,
                        height,
                        ref tail);
                }

                if (count < MinimumComponentSamples)
                    continue;

                float centerX =
                    sumX / count * SampleStep;
                float centerY =
                    sumY / count * SampleStep;
                float score = count;

                if (previousPoint.HasValue)
                {
                    float dx =
                        centerX - previousPoint.Value.X;
                    float dy =
                        centerY - previousPoint.Value.Y;
                    float distance =
                        (float)Math.Sqrt(dx * dx + dy * dy);
                    score -= distance *
                        PreviousPositionWeight;
                }

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

            if (bestCount < MinimumComponentSamples)
                return false;

            tip = new PointF(
                bestSumX / bestCount * SampleStep,
                bestSumY / bestCount * SampleStep);
            bounds = RectangleF.FromLTRB(
                bestMinX * SampleStep,
                bestMinY * SampleStep,
                Math.Min(
                    frame.Width,
                    (bestMaxX + 1) * SampleStep),
                Math.Min(
                    frame.Height,
                    (bestMaxY + 1) * SampleStep));
            previousPoint = tip;
            return true;
        }

        private void EnsureBuffers(int width, int height)
        {
            int requiredLength = width * height;

            if (mask == null ||
                mask.Length < requiredLength)
            {
                mask = new bool[requiredLength];
                queue = new int[requiredLength];
            }

            gridWidth = width;
            gridHeight = height;
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
            int maximum =
                Math.Max(red, Math.Max(green, blue));
            int minimum =
                Math.Min(red, Math.Min(green, blue));
            int chroma = maximum - minimum;

            return red >= 145 &&
                green >= 120 &&
                blue <= 175 &&
                chroma >= 38 &&
                red - blue >= 35 &&
                green - blue >= 28 &&
                Math.Abs(red - green) <= 105;
        }
    }
}
