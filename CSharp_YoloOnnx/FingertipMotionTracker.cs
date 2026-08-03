using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace CSharp_YoloOnnx
{
    /// <summary>
    /// Short-lived inter-frame fingertip tracker. A confirmed landmark
    /// supplies the template and position. Local block matching then fills
    /// camera frames between landmark inferences for at most a brief window.
    /// </summary>
    public sealed class FingertipMotionTracker
    {
        private const int PatchRadius = 7;
        private const int PatchSize = PatchRadius * 2 + 1;
        private const int SearchRadius = 42;
        private const int SearchStep = 3;
        private const float MaximumMeanError = 30f;
        private const float TemplateUpdateFactor = 0.12f;

        private readonly byte[] template =
            new byte[PatchSize * PatchSize];

        private PointF position;
        private PointF velocity;
        private bool initialized;

        public void Reset()
        {
            initialized = false;
            position = PointF.Empty;
            velocity = PointF.Empty;
            Array.Clear(template, 0, template.Length);
        }

        public bool Reset(
            Bitmap frame,
            PointF confirmedPoint)
        {
            Reset();

            if (frame == null ||
                !IsPatchInside(frame, confirmedPoint))
            {
                return false;
            }

            if (!TryReadPatch(
                frame,
                confirmedPoint,
                template))
            {
                return false;
            }

            position = confirmedPoint;
            initialized = true;
            return true;
        }

        public bool TryTrack(
            Bitmap frame,
            out PointF trackedPoint,
            out float confidence)
        {
            trackedPoint = PointF.Empty;
            confidence = 0f;

            if (!initialized || frame == null)
                return false;

            PointF predicted = new PointF(
                position.X + velocity.X,
                position.Y + velocity.Y);
            Point bestXy = Point.Empty;
            float bestError = float.MaxValue;

            Rectangle bounds =
                new Rectangle(
                    0,
                    0,
                    frame.Width,
                    frame.Height);
            BitmapData data = null;

            try
            {
                data = frame.LockBits(
                    bounds,
                    ImageLockMode.ReadOnly,
                    PixelFormat.Format24bppRgb);

                unsafe
                {
                    byte* baseAddress =
                        (byte*)data.Scan0.ToPointer();
                    int minimumX = Math.Max(
                        PatchRadius,
                        (int)Math.Round(predicted.X) -
                        SearchRadius);
                    int maximumX = Math.Min(
                        frame.Width - PatchRadius - 1,
                        (int)Math.Round(predicted.X) +
                        SearchRadius);
                    int minimumY = Math.Max(
                        PatchRadius,
                        (int)Math.Round(predicted.Y) -
                        SearchRadius);
                    int maximumY = Math.Min(
                        frame.Height - PatchRadius - 1,
                        (int)Math.Round(predicted.Y) +
                        SearchRadius);

                    for (int y = minimumY;
                        y <= maximumY;
                        y += SearchStep)
                    {
                        for (int x = minimumX;
                            x <= maximumX;
                            x += SearchStep)
                        {
                            float error =
                                CalculateMeanError(
                                    baseAddress,
                                    data.Stride,
                                    x,
                                    y);

                            if (error < bestError)
                            {
                                bestError = error;
                                bestXy = new Point(x, y);
                            }
                        }
                    }

                    if (bestError > MaximumMeanError ||
                        bestXy == Point.Empty)
                    {
                        return false;
                    }

                    UpdateTemplate(
                        baseAddress,
                        data.Stride,
                        bestXy.X,
                        bestXy.Y);
                }
            }
            catch
            {
                Reset();
                return false;
            }
            finally
            {
                if (data != null)
                    frame.UnlockBits(data);
            }

            PointF next =
                new PointF(bestXy.X, bestXy.Y);
            float movementX =
                next.X - position.X;
            float movementY =
                next.Y - position.Y;

            velocity = new PointF(
                velocity.X * 0.55f +
                movementX * 0.45f,
                velocity.Y * 0.55f +
                movementY * 0.45f);
            position = next;
            trackedPoint = next;
            confidence = Math.Max(
                0f,
                Math.Min(
                    1f,
                    1f -
                    bestError /
                    MaximumMeanError));
            return true;
        }

        private static bool IsPatchInside(
            Bitmap frame,
            PointF point)
        {
            return
                point.X >= PatchRadius &&
                point.Y >= PatchRadius &&
                point.X <
                    frame.Width - PatchRadius &&
                point.Y <
                    frame.Height - PatchRadius;
        }

        private bool TryReadPatch(
            Bitmap frame,
            PointF center,
            byte[] destination)
        {
            Rectangle bounds =
                new Rectangle(
                    0,
                    0,
                    frame.Width,
                    frame.Height);
            BitmapData data = null;

            try
            {
                data = frame.LockBits(
                    bounds,
                    ImageLockMode.ReadOnly,
                    PixelFormat.Format24bppRgb);

                unsafe
                {
                    byte* baseAddress =
                        (byte*)data.Scan0.ToPointer();
                    int centerX =
                        (int)Math.Round(center.X);
                    int centerY =
                        (int)Math.Round(center.Y);
                    int index = 0;

                    for (int offsetY = -PatchRadius;
                        offsetY <= PatchRadius;
                        offsetY++)
                    {
                        byte* row =
                            GetRow(
                                baseAddress,
                                data.Stride,
                                frame.Height,
                                centerY + offsetY);

                        for (int offsetX = -PatchRadius;
                            offsetX <= PatchRadius;
                            offsetX++)
                        {
                            byte* pixel =
                                row +
                                (centerX + offsetX) *
                                3;
                            destination[index++] =
                                ToGray(pixel);
                        }
                    }
                }

                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                if (data != null)
                    frame.UnlockBits(data);
            }
        }

        private unsafe float CalculateMeanError(
            byte* baseAddress,
            int stride,
            int centerX,
            int centerY)
        {
            int index = 0;
            int totalError = 0;

            for (int offsetY = -PatchRadius;
                offsetY <= PatchRadius;
                offsetY++)
            {
                byte* row =
                    baseAddress +
                    (centerY + offsetY) *
                    stride;

                for (int offsetX = -PatchRadius;
                    offsetX <= PatchRadius;
                    offsetX++)
                {
                    byte* pixel =
                        row +
                        (centerX + offsetX) *
                        3;
                    int gray = ToGray(pixel);
                    totalError += Math.Abs(
                        gray - template[index++]);
                }
            }

            return totalError /
                (float)template.Length;
        }

        private unsafe void UpdateTemplate(
            byte* baseAddress,
            int stride,
            int centerX,
            int centerY)
        {
            int index = 0;

            for (int offsetY = -PatchRadius;
                offsetY <= PatchRadius;
                offsetY++)
            {
                byte* row =
                    baseAddress +
                    (centerY + offsetY) *
                    stride;

                for (int offsetX = -PatchRadius;
                    offsetX <= PatchRadius;
                    offsetX++)
                {
                    byte* pixel =
                        row +
                        (centerX + offsetX) *
                        3;
                    float current = template[index];
                    float observed = ToGray(pixel);
                    template[index++] =
                        (byte)Math.Max(
                            0f,
                            Math.Min(
                                255f,
                                current +
                                (observed - current) *
                                TemplateUpdateFactor));
                }
            }
        }

        private static unsafe byte* GetRow(
            byte* baseAddress,
            int stride,
            int height,
            int y)
        {
            return stride >= 0
                ? baseAddress + y * stride
                : baseAddress +
                  (height - 1 - y) *
                  -stride;
        }

        private static unsafe byte ToGray(
            byte* bgr)
        {
            return (byte)(
                (bgr[2] * 77 +
                 bgr[1] * 150 +
                 bgr[0] * 29) >>
                8);
        }
    }
}
