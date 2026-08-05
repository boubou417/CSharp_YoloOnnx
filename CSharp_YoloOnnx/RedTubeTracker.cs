using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace CSharp_YoloOnnx
{
    /// <summary>
    /// Locks a thin red tube to one pose wrist, follows the connected red
    /// path around bends, and returns the path endpoint opposite the grip.
    /// The lock and jump gate prevent background candidates or wrist swaps
    /// from becoming drawing points during a stroke.
    /// </summary>
    public sealed class RedTubeTracker
    {
        private const double ReferencePixelCount = 1600000d;
        private const double HighResolutionPixelCount = 1000000d;
        private const double VeryHighResolutionPixelCount = 4200000d;
        private const float SearchRadius = 650f;
        private const float MaximumWristDistance = 220f;
        private const int MinimumFullResolutionSamples = 32;
        private const int MinimumSampledSamples = 10;
        private const float MinimumTubeSpan = 55f;
        private const float MinimumTubePathLength = 110f;
        private const float MaximumTubeFillRatio = 0.58f;
        private const float MaximumTubeThickness = 40f;
        private const float MinimumStrongRedFraction = 0.05f;
        private const int MinimumStrongRedSamples = 4;
        private const float EndpointBand = 16f;
        private const int PreviousPointMemoryMs = 550;
        private const int ActiveWristLockMs = 1000;
        private const float PreviousPointWeight = 0.70f;
        private const float TrackedBaseJump = 70f;
        private const float TrackedMaximumSpeed = 4200f;
        private const float TrackedMaximumJump = 280f;

        private bool[] mask;
        private bool[] strongMask;
        private int[] queue;
        private int[] pathQueue;
        private int[] pathDistance;
        private int[] componentMarks;
        private int componentGeneration;
        private PointF? previousPoint;
        private DateTime previousPointSeenAt = DateTime.MinValue;
        private int activeWristIndex = -1;
        private DateTime activeWristSeenAt = DateTime.MinValue;

        private sealed class Candidate
        {
            public PointF Point;
            public RectangleF Bounds;
            public float Score = float.MaxValue;
            public int WristIndex = -1;
        }

        public void Reset()
        {
            previousPoint = null;
            previousPointSeenAt = DateTime.MinValue;
            activeWristIndex = -1;
            activeWristSeenAt = DateTime.MinValue;
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
                        0.70f,
                        Math.Min(2.00f, resolutionScale));
                int sampleStep =
                    pixelCount >=
                        VeryHighResolutionPixelCount
                        ? 3
                        : pixelCount >=
                            HighResolutionPixelCount
                            ? 2
                            : 1;
                int minimumSamples =
                    Math.Max(
                        MinimumSampledSamples,
                        MinimumFullResolutionSamples /
                        (sampleStep * sampleStep));
                Candidate best = new Candidate();
                ResetExpiredPoint(now);
                bool leftWristValid =
                    leftWrist.HasValue &&
                    IsInsideFrame(
                        leftWrist.Value,
                        frame.Size);
                bool rightWristValid =
                    rightWrist.HasValue &&
                    IsInsideFrame(
                        rightWrist.Value,
                        frame.Size);
                bool wristLockActive =
                    activeWristIndex >= 0 &&
                    activeWristSeenAt !=
                        DateTime.MinValue &&
                    (now - activeWristSeenAt)
                        .TotalMilliseconds <=
                    ActiveWristLockMs;

                if (wristLockActive)
                {
                    if (activeWristIndex == 0 &&
                        leftWristValid)
                    {
                        FindBestNearWrist(
                            data,
                            frame.Size,
                            bytesPerPixel,
                            leftWrist.Value,
                            0,
                            sampleStep,
                            minimumSamples,
                            resolutionScale,
                            now,
                            best);
                    }
                    else if (activeWristIndex == 1 &&
                        rightWristValid)
                    {
                        FindBestNearWrist(
                            data,
                            frame.Size,
                            bytesPerPixel,
                            rightWrist.Value,
                            1,
                            sampleStep,
                            minimumSamples,
                            resolutionScale,
                            now,
                            best);
                    }
                }
                else
                {
                    if (leftWristValid)
                    {
                        FindBestNearWrist(
                            data,
                            frame.Size,
                            bytesPerPixel,
                            leftWrist.Value,
                            0,
                            sampleStep,
                            minimumSamples,
                            resolutionScale,
                            now,
                            best);
                    }

                    if (rightWristValid)
                    {
                        FindBestNearWrist(
                            data,
                            frame.Size,
                            bytesPerPixel,
                            rightWrist.Value,
                            1,
                            sampleStep,
                            minimumSamples,
                            resolutionScale,
                            now,
                            best);
                    }
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
                activeWristIndex =
                    best.WristIndex;
                activeWristSeenAt = now;
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
            int wristIndex,
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
            Array.Clear(
                strongMask,
                0,
                requiredLength);
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
                    bool strongRed;
                    int pixelIndex =
                        gridY * width + gridX;
                    mask[pixelIndex] =
                        IsRedTube(
                            pixel[2],
                            pixel[1],
                            pixel[0],
                            out strongRed);
                    strongMask[pixelIndex] =
                        mask[pixelIndex] &&
                        strongRed;
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
                int componentMark =
                    NextComponentMark();
                queue[tail++] = index;
                mask[index] = false;
                int count = 0;
                int strongRedCount = 0;
                int minX = width;
                int minY = height;
                int maxX = 0;
                int maxY = 0;
                float nearestSquared =
                    float.MaxValue;
                int nearestIndex = -1;

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
                    componentMarks[current] =
                        componentMark;
                    count++;

                    if (strongMask[current])
                        strongRedCount++;

                    minX = Math.Min(minX, gridX);
                    minY = Math.Min(minY, gridY);
                    maxX = Math.Max(maxX, gridX);
                    maxY = Math.Max(maxY, gridY);
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
                        nearestIndex = current;
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

                int requiredStrongRed =
                    Math.Max(
                        MinimumStrongRedSamples,
                        (int)Math.Ceiling(
                            count *
                            MinimumStrongRedFraction));

                if (strongRedCount <
                    requiredStrongRed)
                {
                    continue;
                }

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

                PointF freeEndpoint;
                float tubePathLength;

                if (nearestIndex < 0 ||
                    !TryFindPathEndpoint(
                        nearestIndex,
                        componentMark,
                        tail,
                        width,
                        height,
                        region,
                        sampleStep,
                        resolutionScale,
                        out freeEndpoint,
                        out tubePathLength))
                {
                    continue;
                }

                if (tubePathLength <
                    MinimumTubePathLength *
                    resolutionScale)
                {
                    continue;
                }

                float sampledArea =
                    count *
                    sampleStep *
                    sampleStep;
                float effectiveThickness =
                    sampledArea /
                    Math.Max(1f, tubePathLength);

                if (effectiveThickness >
                    MaximumTubeThickness *
                    resolutionScale)
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
                        freeEndpoint.X -
                        previousPoint.Value.X;
                    float previousDy =
                        freeEndpoint.Y -
                        previousPoint.Value.Y;
                    float previousDistance =
                        (float)Math.Sqrt(
                            previousDx * previousDx +
                            previousDy * previousDy);
                    double elapsedSeconds =
                        Math.Max(
                            0.001d,
                            (now - previousPointSeenAt)
                                .TotalSeconds);
                    float maximumTrackedJump =
                        Math.Min(
                            TrackedMaximumJump *
                                resolutionScale,
                            TrackedBaseJump *
                                resolutionScale +
                            TrackedMaximumSpeed *
                                resolutionScale *
                            (float)elapsedSeconds);

                    if (previousDistance >
                        maximumTrackedJump)
                    {
                        continue;
                    }

                    temporalPenalty =
                        previousDistance *
                        PreviousPointWeight;
                }

                float score =
                    nearestDistance * 0.80f +
                    temporalPenalty -
                    Math.Min(
                        700f * resolutionScale,
                        tubePathLength) *
                    0.28f +
                    effectiveThickness * 0.60f -
                    Math.Min(
                        260f * resolutionScale,
                        span) *
                    0.05f;

                if (score >= best.Score)
                    continue;

                best.Score = score;
                best.Point = freeEndpoint;
                best.WristIndex = wristIndex;
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

        private bool TryFindPathEndpoint(
            int startIndex,
            int componentMark,
            int componentCount,
            int width,
            int height,
            Rectangle region,
            int sampleStep,
            float resolutionScale,
            out PointF endpoint,
            out float pathLength)
        {
            endpoint = PointF.Empty;
            pathLength = 0f;

            for (int index = 0;
                index < componentCount;
                index++)
            {
                pathDistance[queue[index]] = -1;
            }

            int head = 0;
            int tail = 0;
            pathQueue[tail++] = startIndex;
            pathDistance[startIndex] = 0;
            int maximumDistance = 0;

            while (head < tail)
            {
                int current = pathQueue[head++];
                int gridX = current % width;
                int gridY = current / width;
                int nextDistance =
                    pathDistance[current] + 1;

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

                        EnqueuePathPixel(
                            gridX + offsetX,
                            gridY + offsetY,
                            width,
                            height,
                            componentMark,
                            nextDistance,
                            ref tail);
                    }
                }

                if (pathDistance[current] >
                    maximumDistance)
                {
                    maximumDistance =
                        pathDistance[current];
                }
            }

            if (maximumDistance <= 0)
                return false;

            int endpointBandSteps =
                Math.Max(
                    1,
                    (int)Math.Ceiling(
                        EndpointBand *
                        resolutionScale /
                        sampleStep));
            int endpointThreshold =
                Math.Max(
                    0,
                    maximumDistance -
                    endpointBandSteps);
            double sumX = 0d;
            double sumY = 0d;
            int endpointCount = 0;

            for (int index = 0;
                index < componentCount;
                index++)
            {
                int current = queue[index];

                if (pathDistance[current] <
                    endpointThreshold)
                {
                    continue;
                }

                int gridX = current % width;
                int gridY = current / width;
                sumX +=
                    region.Left +
                    gridX * sampleStep;
                sumY +=
                    region.Top +
                    gridY * sampleStep;
                endpointCount++;
            }

            if (endpointCount <= 0)
                return false;

            endpoint =
                new PointF(
                    (float)(sumX /
                        endpointCount),
                    (float)(sumY /
                        endpointCount));
            pathLength =
                maximumDistance *
                sampleStep;
            return true;
        }

        private void ResetExpiredPoint(
            DateTime now)
        {
            if (previousPointSeenAt !=
                    DateTime.MinValue &&
                (now - previousPointSeenAt)
                    .TotalMilliseconds >
                PreviousPointMemoryMs)
            {
                previousPoint = null;
                previousPointSeenAt =
                    DateTime.MinValue;
            }

            if (activeWristSeenAt !=
                    DateTime.MinValue &&
                (now - activeWristSeenAt)
                    .TotalMilliseconds >
                ActiveWristLockMs)
            {
                activeWristIndex = -1;
                activeWristSeenAt =
                    DateTime.MinValue;
            }
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
            strongMask = new bool[requiredLength];
            queue = new int[requiredLength];
            pathQueue = new int[requiredLength];
            pathDistance = new int[requiredLength];
            componentMarks = new int[requiredLength];
            componentGeneration = 0;
        }

        private int NextComponentMark()
        {
            if (componentGeneration ==
                int.MaxValue)
            {
                Array.Clear(
                    componentMarks,
                    0,
                    componentMarks.Length);
                componentGeneration = 0;
            }

            componentGeneration++;
            return componentGeneration;
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

        private void EnqueuePathPixel(
            int x,
            int y,
            int width,
            int height,
            int componentMark,
            int distance,
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

            if (componentMarks[index] !=
                    componentMark ||
                pathDistance[index] >= 0)
            {
                return;
            }

            pathDistance[index] = distance;
            pathQueue[tail++] = index;
        }

        private static bool IsRedTube(
            byte red,
            byte green,
            byte blue,
            out bool strongRed)
        {
            strongRed = false;
            float r = red / 255f;
            float g = green / 255f;
            float b = blue / 255f;
            float maximum =
                Math.Max(r, Math.Max(g, b));
            float minimum =
                Math.Min(r, Math.Min(g, b));
            float chroma = maximum - minimum;

            if (maximum < 0.24f ||
                chroma < 0.12f)
            {
                return false;
            }

            float saturation =
                maximum <= 0f
                    ? 0f
                    : chroma / maximum;

            if (saturation < 0.44f)
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

            bool possibleRed =
                hue <= 30f ||
                hue >= 346f;

            if (!possibleRed)
                return false;

            strongRed =
                maximum >= 0.30f &&
                chroma >= 0.17f &&
                saturation >= 0.54f &&
                (hue <= 26f ||
                 hue >= 350f);
            return true;
        }
    }
}
