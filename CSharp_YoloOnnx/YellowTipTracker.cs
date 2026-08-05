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
        private const float PreviousPositionWeight = 0.08f;
        private const int PendingJumpMemoryMs = 140;
        private const float SuspiciousJumpDiagonalRatio = 0.035f;
        private const float PredictionErrorDiagonalRatio = 0.055f;
        private const float PendingMatchDiagonalRatio = 0.040f;
        private const float MaximumConfirmedStepDiagonalRatio = 0.20f;
        private const int LockedYellowLossMs = 450;
        private const int YellowOnlyPairGraceMs = 120;
        private const float YellowOnlyPredictionDiagonalRatio = 0.025f;
        private const float RedSearchRadius = 60f;
        private const float RedContactDistance = 24f;
        private const int MinimumRedFullFrameSamples = 4;
        private const int MinimumRedRoiSamples = 12;
        private const float MinimumRedTubeSpan = 24f;
        private const float MaximumRedTubeFillRatio = 0.62f;
        private const float MinimumRedTubeElongation = 1.8f;
        private const float MinimumRedEndpointAlignment = 0.45f;
        private const float MaximumYellowComponentSpan = 180f;

        private bool[] mask;
        private int[] queue;
        private bool[] redMask;
        private int[] redQueue;
        private PointF? previousPoint;
        private PointF velocityPixelsPerMillisecond;
        private DateTime previousPointSeenAt = DateTime.MinValue;
        private DateTime lastRedYellowPairSeenAt = DateTime.MinValue;
        private PointF? pendingJumpPoint;
        private DateTime pendingJumpSeenAt = DateTime.MinValue;
        private bool markerLocked;
        private DateTime lockedYellowMissingSince =
            DateTime.MinValue;

        public bool LastDetectionStartedNewLock
        {
            get;
            private set;
        }

        public bool LastDetectionHasRedSupport
        {
            get;
            private set;
        }

        public void Reset()
        {
            ReleaseMarkerLock();
            LastDetectionStartedNewLock = false;
            LastDetectionHasRedSupport = false;
        }

        public unsafe bool TryDetect(
            Bitmap frame,
            out PointF tip,
            out RectangleF bounds)
        {
            tip = PointF.Empty;
            bounds = RectangleF.Empty;
            LastDetectionStartedNewLock = false;
            LastDetectionHasRedSupport = false;

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
                float resolutionScale =
                    GetResolutionScale(frame.Size);
                // High-megapixel wide-angle cameras show more pixels but
                // do not necessarily make the physical marker larger.
                // Bound geometric expansion until the observed component
                // size can constrain red/yellow pairing directly.
                float geometryScale =
                    Math.Min(1.25f, resolutionScale);
                bool hasPrediction =
                    previousPoint.HasValue &&
                    previousPointSeenAt != DateTime.MinValue &&
                    (now - previousPointSeenAt)
                        .TotalMilliseconds <=
                    PredictionMemoryMs;

                if (markerLocked && hasPrediction)
                {
                    double elapsedMilliseconds =
                        Math.Max(
                            1d,
                            (now - previousPointSeenAt)
                                .TotalMilliseconds);
                    PointF prediction = new PointF(
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
                    float minimumRoiRadius =
                        MinimumRoiRadius *
                        geometryScale;
                    float maximumRoiRadius =
                        MaximumRoiRadius *
                        geometryScale;
                    float radius =
                        Math.Max(
                            minimumRoiRadius,
                            Math.Min(
                                maximumRoiRadius,
                                minimumRoiRadius +
                                speed *
                                (float)elapsedMilliseconds *
                                RoiMotionExpansion));
                    Rectangle roi =
                        IntersectWithFrame(
                            prediction,
                            radius,
                            frame.Size);
                    bool hasRedSupport;

                    if (TryFindYellowComponent(
                        data,
                        frame.Size,
                        bytesPerPixel,
                        roi,
                        RoiSampleStep,
                        MinimumRoiSamples,
                        prediction,
                        geometryScale,
                        true,
                        now,
                        out tip,
                        out bounds,
                        out hasRedSupport) &&
                        CanUseLockedCandidate(
                            tip,
                            bounds,
                            prediction,
                            hasRedSupport,
                            now,
                            frame.Size) &&
                        AcceptMotionCandidate(
                            tip,
                            now,
                            frame.Size,
                            hasRedSupport))
                    {
                        ConfirmMarkerLock(
                            hasRedSupport,
                            now);
                        LastDetectionHasRedSupport =
                            hasRedSupport;
                        UpdateMotion(tip, now);
                        return true;
                    }

                    tip = PointF.Empty;
                    bounds = RectangleF.Empty;

                    if (ShouldKeepMarkerLock(now))
                        return false;

                    ReleaseMarkerLock();
                }
                else if (markerLocked)
                {
                    ReleaseMarkerLock();
                }

                // Searching and true-loss reacquisition remain strict:
                // a compact yellow marker must have connected red-tube
                // support before it can establish a new lock.
                bool reacquiredWithRedSupport;

                if (TryFindYellowComponent(
                    data,
                    frame.Size,
                    bytesPerPixel,
                    frameBounds,
                    FullFrameSampleStep,
                    MinimumFullFrameSamples,
                    PointF.Empty,
                    geometryScale,
                    false,
                    now,
                    out tip,
                    out bounds,
                    out reacquiredWithRedSupport) &&
                    AcceptMotionCandidate(
                        tip,
                        now,
                        frame.Size,
                        reacquiredWithRedSupport))
                {
                    ConfirmMarkerLock(
                        reacquiredWithRedSupport,
                        now);
                    LastDetectionStartedNewLock = true;
                    LastDetectionHasRedSupport =
                        reacquiredWithRedSupport;
                    UpdateMotion(tip, now);
                    return true;
                }

                tip = PointF.Empty;
                bounds = RectangleF.Empty;
                return false;
            }
            finally
            {
                frame.UnlockBits(data);
            }
        }

        private void ConfirmMarkerLock(
            bool hasRedSupport,
            DateTime now)
        {
            markerLocked = true;
            lockedYellowMissingSince =
                DateTime.MinValue;

            if (hasRedSupport)
                lastRedYellowPairSeenAt = now;
        }

        private bool CanUseLockedCandidate(
            PointF candidate,
            RectangleF candidateBounds,
            PointF prediction,
            bool hasRedSupport,
            DateTime now,
            Size frameSize)
        {
            if (hasRedSupport)
                return true;

            if (lastRedYellowPairSeenAt ==
                    DateTime.MinValue ||
                (now - lastRedYellowPairSeenAt)
                    .TotalMilliseconds >
                YellowOnlyPairGraceMs)
            {
                return false;
            }

            float markerSpan =
                Math.Max(
                    candidateBounds.Width,
                    candidateBounds.Height);
            float edgeMargin =
                Math.Max(8f, markerSpan * 0.75f);

            // A marker leaving the image must not hand its lock to a
            // similarly coloured object along the frame boundary.
            if (candidate.X <= edgeMargin ||
                candidate.Y <= edgeMargin ||
                candidate.X >=
                    frameSize.Width - edgeMargin ||
                candidate.Y >=
                    frameSize.Height - edgeMargin)
            {
                return false;
            }

            float dx = candidate.X - prediction.X;
            float dy = candidate.Y - prediction.Y;
            float predictionError =
                (float)Math.Sqrt(dx * dx + dy * dy);
            float diagonal =
                (float)Math.Sqrt(
                    frameSize.Width * frameSize.Width +
                    frameSize.Height * frameSize.Height);
            float maximumPredictionError =
                Math.Max(
                    32f,
                    Math.Min(
                        diagonal *
                            YellowOnlyPredictionDiagonalRatio,
                        Math.Max(32f, markerSpan * 3f)));

            return predictionError <=
                maximumPredictionError;
        }

        private bool ShouldKeepMarkerLock(
            DateTime now)
        {
            if (!markerLocked)
                return false;

            if (lockedYellowMissingSince ==
                DateTime.MinValue)
            {
                lockedYellowMissingSince = now;
                return true;
            }

            return (now - lockedYellowMissingSince)
                .TotalMilliseconds <
                LockedYellowLossMs;
        }

        private void ReleaseMarkerLock()
        {
            markerLocked = false;
            lockedYellowMissingSince =
                DateTime.MinValue;
            previousPoint = null;
            velocityPixelsPerMillisecond =
                PointF.Empty;
            previousPointSeenAt =
                DateTime.MinValue;
            lastRedYellowPairSeenAt =
                DateTime.MinValue;
            ClearPendingJump();
        }

        private unsafe bool TryFindYellowComponent(
            BitmapData data,
            Size frameSize,
            int bytesPerPixel,
            Rectangle searchRegion,
            int sampleStep,
            int minimumSamples,
            PointF referencePoint,
            float resolutionScale,
            bool allowYellowOnly,
            DateTime now,
            out PointF tip,
            out RectangleF bounds,
            out bool selectedHasRedSupport)
        {
            tip = PointF.Empty;
            bounds = RectangleF.Empty;
            selectedHasRedSupport = false;
            minimumSamples =
                ScaleSampleCount(
                    minimumSamples,
                    resolutionScale);

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
            bool bestHasRedSupport = false;

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

                float componentWidth =
                    (maxX - minX + 1) * sampleStep;
                float componentHeight =
                    (maxY - minY + 1) * sampleStep;

                // Large yellow/orange surfaces are background, not the
                // compact tape marker fitted to the red tube.
                float maximumYellowSpan =
                    MaximumYellowComponentSpan *
                    resolutionScale;

                if (componentWidth > maximumYellowSpan ||
                    componentHeight > maximumYellowSpan)
                {
                    continue;
                }

                float centerX =
                    searchRegion.Left +
                    sumX / count *
                    sampleStep;
                float centerY =
                    searchRegion.Top +
                    sumY / count *
                    sampleStep;
                RectangleF componentBounds =
                    RectangleF.FromLTRB(
                        searchRegion.Left +
                        minX * sampleStep,
                        searchRegion.Top +
                        minY * sampleStep,
                        Math.Min(
                            frameSize.Width,
                            searchRegion.Left +
                            (maxX + 1) * sampleStep),
                        Math.Min(
                            frameSize.Height,
                            searchRegion.Top +
                            (maxY + 1) * sampleStep));
                bool hasRedSupport =
                    HasAdjacentRedTube(
                        data,
                        frameSize,
                        bytesPerPixel,
                        componentBounds,
                        new PointF(centerX, centerY),
                        sampleStep == RoiSampleStep
                            ? MinimumRedRoiSamples
                            : MinimumRedFullFrameSamples,
                        sampleStep,
                        resolutionScale);

                if (!hasRedSupport && !allowYellowOnly)
                    continue;

                float dx =
                    centerX - referencePoint.X;
                float dy =
                    centerY - referencePoint.Y;
                float distance =
                    (float)Math.Sqrt(dx * dx + dy * dy);
                bool hasReference =
                    referencePoint.X != 0f ||
                    referencePoint.Y != 0f;
                float areaScale =
                    Math.Max(
                        0.25f,
                        resolutionScale *
                        resolutionScale);
                float normalizedCount =
                    count / areaScale;
                float normalizedDistance =
                    distance /
                    Math.Max(0.5f, resolutionScale);
                float score =
                    normalizedCount -
                    normalizedDistance *
                    (hasReference
                        ? PreviousPositionWeight
                        : 0f) +
                    (hasRedSupport
                        ? Math.Max(
                            12f,
                            normalizedCount * 0.25f)
                        : -12f);

                bool improvesPairClass =
                    hasRedSupport &&
                    !bestHasRedSupport;
                bool losesPairClass =
                    !hasRedSupport &&
                    bestHasRedSupport;

                // A geometrically valid red/yellow pair always wins over
                // a larger yellow-only background component.
                if (losesPairClass ||
                    (!improvesPairClass &&
                     score <= bestScore))
                {
                    continue;
                }

                bestScore = score;
                bestCount = count;
                bestSumX = sumX;
                bestSumY = sumY;
                bestMinX = minX;
                bestMinY = minY;
                bestMaxX = maxX;
                bestMaxY = maxY;
                bestHasRedSupport = hasRedSupport;
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
            selectedHasRedSupport =
                bestHasRedSupport;
            return true;
        }

        private unsafe bool HasAdjacentRedTube(
            BitmapData data,
            Size frameSize,
            int bytesPerPixel,
            RectangleF yellowBounds,
            PointF yellowCenter,
            int minimumSamples,
            int sampleStep,
            float resolutionScale)
        {
            minimumSamples =
                ScaleSampleCount(
                    minimumSamples,
                    resolutionScale);
            float yellowSpan =
                Math.Max(
                    yellowBounds.Width,
                    yellowBounds.Height);
            float redSearchRadius =
                Math.Max(
                    48f,
                    Math.Min(
                        RedSearchRadius *
                            resolutionScale,
                        Math.Max(48f, yellowSpan * 4f)));
            float redContactDistance =
                Math.Max(
                    6f,
                    Math.Min(
                        RedContactDistance *
                            resolutionScale,
                        Math.Max(6f, yellowSpan * 0.85f)));
            float minimumRedTubeSpan =
                Math.Max(
                    18f,
                    Math.Min(
                        MinimumRedTubeSpan *
                            resolutionScale,
                        Math.Max(18f, yellowSpan * 2.2f)));
            Rectangle region = Rectangle.Intersect(
                new Rectangle(0, 0, frameSize.Width, frameSize.Height),
                Rectangle.FromLTRB(
                    (int)Math.Floor(yellowBounds.Left - redSearchRadius),
                    (int)Math.Floor(yellowBounds.Top - redSearchRadius),
                    (int)Math.Ceiling(yellowBounds.Right + redSearchRadius),
                    (int)Math.Ceiling(yellowBounds.Bottom + redSearchRadius)));

            if (region.Width <= 0 || region.Height <= 0)
                return false;

            int width =
                (region.Width + sampleStep - 1) / sampleStep;
            int height =
                (region.Height + sampleStep - 1) / sampleStep;
            int requiredLength = width * height;
            EnsureRedBuffers(requiredLength);
            Array.Clear(redMask, 0, requiredLength);
            byte* scan0 = (byte*)data.Scan0.ToPointer();

            for (int gy = 0; gy < height; gy++)
            {
                int sourceY = Math.Min(
                    region.Bottom - 1,
                    region.Top + gy * sampleStep);
                byte* row =
                    data.Stride >= 0
                        ? scan0 + sourceY * data.Stride
                        : scan0 +
                          (frameSize.Height - 1 - sourceY) *
                          -data.Stride;

                for (int gx = 0; gx < width; gx++)
                {
                    int sourceX = Math.Min(
                        region.Right - 1,
                        region.Left + gx * sampleStep);
                    byte* pixel =
                        row + sourceX * bytesPerPixel;
                    redMask[gy * width + gx] =
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
                if (!redMask[index])
                    continue;

                int head = 0;
                int tail = 0;
                redQueue[tail++] = index;
                redMask[index] = false;
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
                float nearestSquared = float.MaxValue;
                float farthestSquared = 0f;

                while (head < tail)
                {
                    int current = redQueue[head++];
                    int x = current % width;
                    int y = current / width;
                    int sourceX =
                        region.Left + x * sampleStep;
                    int sourceY =
                        region.Top + y * sampleStep;
                    count++;
                    minX = Math.Min(minX, x);
                    minY = Math.Min(minY, y);
                    maxX = Math.Max(maxX, x);
                    maxY = Math.Max(maxY, y);
                    sumX += sourceX;
                    sumY += sourceY;
                    sumXX += sourceX * sourceX;
                    sumYY += sourceY * sourceY;
                    sumXY += sourceX * sourceY;
                    float dx = sourceX - yellowCenter.X;
                    float dy = sourceY - yellowCenter.Y;
                    float distanceSquared = dx * dx + dy * dy;
                    nearestSquared =
                        Math.Min(nearestSquared, distanceSquared);
                    farthestSquared =
                        Math.Max(farthestSquared, distanceSquared);

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
                                x + offsetX,
                                y + offsetY,
                                width,
                                height,
                                ref tail);
                        }
                    }
                }

                if (count < minimumSamples ||
                    nearestSquared >
                    redContactDistance * redContactDistance ||
                    farthestSquared <
                    minimumRedTubeSpan * minimumRedTubeSpan)
                {
                    continue;
                }

                int componentWidth = maxX - minX + 1;
                int componentHeight = maxY - minY + 1;
                float fillRatio =
                    count /
                    (float)(componentWidth * componentHeight);

                if (fillRatio > MaximumRedTubeFillRatio)
                    continue;

                double meanX = sumX / count;
                double meanY = sumY / count;
                double covarianceXX =
                    sumXX / count - meanX * meanX;
                double covarianceYY =
                    sumYY / count - meanY * meanY;
                double covarianceXY =
                    sumXY / count - meanX * meanY;
                double trace = covarianceXX + covarianceYY;
                double determinant =
                    covarianceXX * covarianceYY -
                    covarianceXY * covarianceXY;
                double discriminant =
                    Math.Sqrt(
                        Math.Max(
                            0d,
                            trace * trace * 0.25d -
                            determinant));
                double major =
                    trace * 0.5d + discriminant;
                double minor =
                    Math.Max(
                        0.25d,
                        trace * 0.5d - discriminant);
                double elongation = major / minor;

                if (elongation < MinimumRedTubeElongation)
                    continue;

                double axisX;
                double axisY;

                if (Math.Abs(covarianceXY) > 0.000001d)
                {
                    axisX = major - covarianceYY;
                    axisY = covarianceXY;
                }
                else if (covarianceXX >= covarianceYY)
                {
                    axisX = 1d;
                    axisY = 0d;
                }
                else
                {
                    axisX = 0d;
                    axisY = 1d;
                }

                double axisLength =
                    Math.Sqrt(
                        axisX * axisX +
                        axisY * axisY);

                if (axisLength <= 0.000001d)
                    continue;

                axisX /= axisLength;
                axisY /= axisLength;
                double minimumProjection =
                    double.MaxValue;
                double maximumProjection =
                    double.MinValue;

                for (int queueIndex = 0;
                    queueIndex < tail;
                    queueIndex++)
                {
                    int redIndex =
                        redQueue[queueIndex];
                    int redX =
                        region.Left +
                        redIndex % width * sampleStep;
                    int redY =
                        region.Top +
                        redIndex / width * sampleStep;
                    double projection =
                        (redX - meanX) * axisX +
                        (redY - meanY) * axisY;
                    minimumProjection =
                        Math.Min(
                            minimumProjection,
                            projection);
                    maximumProjection =
                        Math.Max(
                            maximumProjection,
                            projection);
                }

                double yellowFromMeanX =
                    yellowCenter.X - meanX;
                double yellowFromMeanY =
                    yellowCenter.Y - meanY;
                double yellowCenterDistance =
                    Math.Sqrt(
                        yellowFromMeanX * yellowFromMeanX +
                        yellowFromMeanY * yellowFromMeanY);
                double yellowProjection =
                    yellowFromMeanX * axisX +
                    yellowFromMeanY * axisY;
                double endpointTolerance =
                    Math.Max(
                        redContactDistance,
                        yellowSpan * 0.75f);
                bool nearPositiveEndpoint =
                    yellowProjection >=
                    maximumProjection -
                    endpointTolerance;
                bool nearNegativeEndpoint =
                    yellowProjection <=
                    minimumProjection +
                    endpointTolerance;
                double endpointAlignment =
                    yellowCenterDistance <= 0.000001d
                        ? 0d
                        : Math.Abs(yellowProjection) /
                          yellowCenterDistance;

                // The yellow component must sit at an end of the long red
                // component. Merely having unrelated warm pixels nearby is
                // not enough to establish or refresh the marker lock.
                if ((!nearPositiveEndpoint &&
                     !nearNegativeEndpoint) ||
                    endpointAlignment <
                    MinimumRedEndpointAlignment)
                {
                    continue;
                }

                return true;
            }

            return false;
        }

        private void EnsureRedBuffers(int requiredLength)
        {
            if (redMask == null ||
                redMask.Length < requiredLength)
            {
                redMask = new bool[requiredLength];
                redQueue = new int[requiredLength];
            }
        }

        private void EnqueueIfRed(
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

            if (!redMask[index])
                return;

            redMask[index] = false;
            redQueue[tail++] = index;
        }

        private bool AcceptMotionCandidate(
            PointF candidate,
            DateTime now,
            Size frameSize,
            bool hasRedSupport)
        {
            if (!previousPoint.HasValue ||
                previousPointSeenAt == DateTime.MinValue ||
                (now - previousPointSeenAt)
                    .TotalMilliseconds >
                PredictionMemoryMs)
            {
                ClearPendingJump();
                return true;
            }

            float diagonal =
                (float)Math.Sqrt(
                    frameSize.Width * frameSize.Width +
                    frameSize.Height * frameSize.Height);
            double elapsedMilliseconds =
                Math.Max(
                    1d,
                    (now - previousPointSeenAt)
                        .TotalMilliseconds);
            float dx =
                candidate.X - previousPoint.Value.X;
            float dy =
                candidate.Y - previousPoint.Value.Y;
            float distance =
                (float)Math.Sqrt(dx * dx + dy * dy);
            PointF expected = new PointF(
                previousPoint.Value.X +
                velocityPixelsPerMillisecond.X *
                (float)elapsedMilliseconds,
                previousPoint.Value.Y +
                velocityPixelsPerMillisecond.Y *
                (float)elapsedMilliseconds);
            float expectedDx = candidate.X - expected.X;
            float expectedDy = candidate.Y - expected.Y;
            float predictionError =
                (float)Math.Sqrt(
                    expectedDx * expectedDx +
                    expectedDy * expectedDy);
            float previousSpeed =
                (float)Math.Sqrt(
                    velocityPixelsPerMillisecond.X *
                    velocityPixelsPerMillisecond.X +
                    velocityPixelsPerMillisecond.Y *
                    velocityPixelsPerMillisecond.Y);
            float elapsedScale =
                Math.Max(
                    1f,
                    Math.Min(
                        2.5f,
                        (float)elapsedMilliseconds / 33f));
            float suspiciousDistance =
                Math.Max(
                    48f,
                    diagonal *
                    SuspiciousJumpDiagonalRatio *
                    elapsedScale);
            float predictionAllowance =
                Math.Max(
                    diagonal *
                    PredictionErrorDiagonalRatio,
                    previousSpeed *
                    (float)elapsedMilliseconds *
                    1.35f +
                    diagonal * 0.012f);
            bool reversesDirection = false;

            if (previousSpeed > 0.10f &&
                distance > 1f)
            {
                float dot =
                    dx *
                    velocityPixelsPerMillisecond.X +
                    dy *
                    velocityPixelsPerMillisecond.Y;
                reversesDirection =
                    dot /
                    (distance * previousSpeed) <
                    -0.25f;
            }

            bool suspicious =
                distance > suspiciousDistance &&
                (predictionError > predictionAllowance ||
                 reversesDirection);

            if (!suspicious)
            {
                ClearPendingJump();
                return true;
            }

            // Yellow-only tracking is a very short blur/occlusion grace.
            // It may never authorize a large jump to a background object.
            if (!hasRedSupport)
            {
                ClearPendingJump();
                return false;
            }

            if (pendingJumpPoint.HasValue &&
                pendingJumpSeenAt != DateTime.MinValue &&
                (now - pendingJumpSeenAt)
                    .TotalMilliseconds <=
                PendingJumpMemoryMs)
            {
                float pendingDx =
                    candidate.X - pendingJumpPoint.Value.X;
                float pendingDy =
                    candidate.Y - pendingJumpPoint.Value.Y;
                float pendingStep =
                    (float)Math.Sqrt(
                        pendingDx * pendingDx +
                        pendingDy * pendingDy);
                float firstDx =
                    pendingJumpPoint.Value.X -
                    previousPoint.Value.X;
                float firstDy =
                    pendingJumpPoint.Value.Y -
                    previousPoint.Value.Y;
                float firstStep =
                    (float)Math.Sqrt(
                        firstDx * firstDx +
                        firstDy * firstDy);
                bool matchesPending =
                    pendingStep <=
                    Math.Max(
                        42f,
                        diagonal *
                        PendingMatchDiagonalRatio);
                bool continuesDirection = false;

                if (firstStep > 1f &&
                    pendingStep > 1f &&
                    pendingStep <=
                    diagonal *
                    MaximumConfirmedStepDiagonalRatio)
                {
                    float directionCosine =
                        (firstDx * pendingDx +
                         firstDy * pendingDy) /
                        (firstStep * pendingStep);
                    continuesDirection =
                        directionCosine >= 0.15f;
                }

                if (matchesPending ||
                    continuesDirection)
                {
                    ClearPendingJump();
                    return true;
                }
            }

            pendingJumpPoint = candidate;
            pendingJumpSeenAt = now;
            return false;
        }

        private void ClearPendingJump()
        {
            pendingJumpPoint = null;
            pendingJumpSeenAt = DateTime.MinValue;
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

        private static float GetResolutionScale(
            Size frameSize)
        {
            const double referencePixelCount = 1600000d;
            double pixelCount =
                Math.Max(
                    1d,
                    (double)frameSize.Width *
                    frameSize.Height);
            float scale =
                (float)Math.Sqrt(
                    pixelCount /
                    referencePixelCount);

            return Math.Max(
                0.75f,
                Math.Min(2.5f, scale));
        }

        private static int ScaleSampleCount(
            int baseCount,
            float resolutionScale)
        {
            // Saturated marker pixels do not reliably grow with total
            // camera megapixels because lens, field of view, exposure,
            // motion blur, and binning all change the visible component.
            // Keep the proven 1.6 MP sample counts while scaling only
            // geometric distances and search regions.
            return Math.Max(3, baseCount);
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

        private static bool IsRedTube(
            byte red,
            byte green,
            byte blue)
        {
            float r = red / 255f;
            float g = green / 255f;
            float b = blue / 255f;
            float maximum = Math.Max(r, Math.Max(g, b));
            float minimum = Math.Min(r, Math.Min(g, b));
            float chroma = maximum - minimum;

            if (maximum < 0.32f || chroma < 0.18f)
                return false;

            float saturation =
                maximum <= 0f ? 0f : chroma / maximum;

            if (saturation < 0.48f)
                return false;

            float hue;

            if (maximum == r)
                hue = 60f * (((g - b) / chroma) % 6f);
            else if (maximum == g)
                hue = 60f * (((b - r) / chroma) + 2f);
            else
                hue = 60f * (((r - g) / chroma) + 4f);

            if (hue < 0f)
                hue += 360f;

            // Include the orange-red flexible tube seen under warm
            // exhibition lighting, but exclude yellow hues.
            return hue <= 22f || hue >= 350f;
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

            return hue >= 42f &&
                hue <= 76f;
        }
    }
}
