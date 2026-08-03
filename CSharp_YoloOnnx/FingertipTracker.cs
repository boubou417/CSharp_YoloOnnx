using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using Emgu.TF.Lite;

namespace CSharp_YoloOnnx
{
    public sealed class FingertipResult
    {
        public PointF IndexTip { get; set; }
        public PointF ThumbTip { get; set; }
        public PointF Wrist { get; set; }
        public float PinchRatio { get; set; }
        public float OpenPalmScore { get; set; }
        public float HandPresence { get; set; }
        public RectangleF SearchRegion { get; set; }
    }

    /// <summary>
    /// Runs the MediaPipe hand-landmark model on a pose-guided crop around a
    /// wrist. YOLO supplies the wrist and elbow, so a separate palm
    /// detector is not required for this drawing workflow.
    /// </summary>
    public sealed class FingertipTracker : IDisposable
    {
        private const int ModelInputSize = 224;
        private const int LandmarkValueCount = 21 * 3;
        private const int WristLandmarkIndex = 0;
        private const int ThumbTipLandmarkIndex = 4;
        private const int IndexMcpLandmarkIndex = 5;
        private const int IndexPipLandmarkIndex = 6;
        private const int IndexTipLandmarkIndex = 8;
        private const int MiddleMcpLandmarkIndex = 9;
        private const int MiddlePipLandmarkIndex = 10;
        private const int MiddleTipLandmarkIndex = 12;
        private const int RingPipLandmarkIndex = 14;
        private const int RingTipLandmarkIndex = 16;
        private const int PinkyMcpLandmarkIndex = 17;
        private const int PinkyPipLandmarkIndex = 18;
        private const int PinkyTipLandmarkIndex = 20;
        private const float MinimumHandPresence = 0.35f;
        private const float ExtendedFingerDistanceRatio = 1.12f;
        private const float CropSizeFromForearm = 2.2f;
        private const float CropCenterFromWrist = 0.35f;
        private const float MaximumWristMismatchRatio = 0.35f;

        private readonly object syncRoot = new object();
        private readonly float[] inputBuffer =
            new float[ModelInputSize * ModelInputSize * 3];

        private FlatBufferModel model;
        private Interpreter interpreter;
        private Tensor inputTensor;
        private Tensor[] outputTensors;
        private bool disposed;

        public FingertipTracker(string modelPath)
        {
            if (string.IsNullOrWhiteSpace(modelPath))
                throw new ArgumentException("A hand-landmark model path is required.", nameof(modelPath));

            if (!File.Exists(modelPath))
                throw new FileNotFoundException("Hand-landmark model was not found.", modelPath);

            model = new FlatBufferModel(modelPath);

            if (!model.CheckModelIdentifier())
                throw new InvalidDataException("The hand-landmark model identifier is invalid.");

            interpreter = new Interpreter(model);
            interpreter.SetNumThreads(2);

            if (interpreter.AllocateTensors() != Status.Ok)
                throw new InvalidOperationException("TensorFlow Lite could not allocate tensors.");

            Tensor[] inputs = interpreter.Inputs;
            if (inputs.Length != 1)
                throw new InvalidDataException("The hand-landmark model must have one input tensor.");

            inputTensor = inputs[0];
            outputTensors = interpreter.Outputs;

            ValidateInputTensor(inputTensor);

            if (outputTensors.Length < 2)
                throw new InvalidDataException("The hand-landmark model outputs are incomplete.");
        }

        public bool TryDetect(
            Bitmap frame,
            PointF wrist,
            PointF elbow,
            out FingertipResult result)
        {
            return TryDetect(
                frame,
                wrist,
                elbow,
                MinimumHandPresence,
                out result);
        }

        public bool TryDetect(
            Bitmap frame,
            PointF wrist,
            PointF elbow,
            float minimumHandPresence,
            out FingertipResult result)
        {
            return TryDetect(
                frame,
                wrist,
                elbow,
                minimumHandPresence,
                1f,
                out result);
        }

        public bool TryDetect(
            Bitmap frame,
            PointF wrist,
            PointF elbow,
            float minimumHandPresence,
            float cropScale,
            out FingertipResult result)
        {
            result = null;

            if (frame == null)
                return false;

            lock (syncRoot)
            {
                if (disposed)
                    return false;

                CropTransform transform;
                if (!TryCreateCropTransform(
                    frame.Size,
                    wrist,
                    elbow,
                    cropScale,
                    out transform))
                {
                    return false;
                }

                FillInputTensor(frame, transform);
                Marshal.Copy(inputBuffer, 0, inputTensor.DataPointer, inputBuffer.Length);

                if (interpreter.Invoke() != Status.Ok)
                    return false;

                float[] landmarks;
                float handPresence;

                if (!TryReadOutputs(out landmarks, out handPresence) ||
                    handPresence < minimumHandPresence)
                {
                    return false;
                }

                PointF modelWrist = ReadLandmark(landmarks, WristLandmarkIndex);
                PointF detectedWrist = transform.MapToFrame(modelWrist);

                float wristDx = detectedWrist.X - wrist.X;
                float wristDy = detectedWrist.Y - wrist.Y;
                float maximumMismatch =
                    transform.SideLength * MaximumWristMismatchRatio;

                if (wristDx * wristDx + wristDy * wristDy >
                    maximumMismatch * maximumMismatch)
                {
                    return false;
                }

                PointF modelIndexTip = ReadLandmark(
                    landmarks,
                    IndexTipLandmarkIndex);
                PointF modelThumbTip = ReadLandmark(
                    landmarks,
                    ThumbTipLandmarkIndex);
                PointF modelMiddleMcp = ReadLandmark(
                    landmarks,
                    MiddleMcpLandmarkIndex);
                PointF modelIndexMcp = ReadLandmark(
                    landmarks,
                    IndexMcpLandmarkIndex);
                PointF modelPinkyMcp = ReadLandmark(
                    landmarks,
                    PinkyMcpLandmarkIndex);

                float palmLength = Distance(
                    modelWrist,
                    modelMiddleMcp);
                float palmWidth = Distance(
                    modelIndexMcp,
                    modelPinkyMcp);
                float handScale = Math.Max(palmLength, palmWidth);

                if (handScale < 1f)
                    return false;

                float pinchRatio =
                    Distance(modelThumbTip, modelIndexTip) /
                    handScale;
                int extendedFingerCount = 0;

                if (IsFingerExtended(
                    landmarks,
                    modelWrist,
                    IndexPipLandmarkIndex,
                    IndexTipLandmarkIndex))
                {
                    extendedFingerCount++;
                }

                if (IsFingerExtended(
                    landmarks,
                    modelWrist,
                    MiddlePipLandmarkIndex,
                    MiddleTipLandmarkIndex))
                {
                    extendedFingerCount++;
                }

                if (IsFingerExtended(
                    landmarks,
                    modelWrist,
                    RingPipLandmarkIndex,
                    RingTipLandmarkIndex))
                {
                    extendedFingerCount++;
                }

                if (IsFingerExtended(
                    landmarks,
                    modelWrist,
                    PinkyPipLandmarkIndex,
                    PinkyTipLandmarkIndex))
                {
                    extendedFingerCount++;
                }

                float openPalmScore =
                    extendedFingerCount / 4f;
                PointF indexTip = transform.MapToFrame(modelIndexTip);
                PointF thumbTip = transform.MapToFrame(modelThumbTip);

                if (indexTip.X < 0f ||
                    indexTip.Y < 0f ||
                    indexTip.X >= frame.Width ||
                    indexTip.Y >= frame.Height ||
                    thumbTip.X < 0f ||
                    thumbTip.Y < 0f ||
                    thumbTip.X >= frame.Width ||
                    thumbTip.Y >= frame.Height)
                {
                    return false;
                }

                result = new FingertipResult
                {
                    IndexTip = indexTip,
                    ThumbTip = thumbTip,
                    Wrist = detectedWrist,
                    PinchRatio = pinchRatio,
                    OpenPalmScore = openPalmScore,
                    HandPresence = handPresence,
                    SearchRegion = transform.AxisAlignedBounds
                };

                return true;
            }
        }

        private static bool IsFingerExtended(
            float[] landmarks,
            PointF wrist,
            int pipIndex,
            int tipIndex)
        {
            PointF pip = ReadLandmark(landmarks, pipIndex);
            PointF tip = ReadLandmark(landmarks, tipIndex);

            return Distance(wrist, tip) >=
                Distance(wrist, pip) *
                ExtendedFingerDistanceRatio;
        }

        private static void ValidateInputTensor(Tensor tensor)
        {
            if (tensor.Type != DataType.Float32)
                throw new InvalidDataException("The hand-landmark model input must be Float32.");

            int[] dimensions = tensor.Dims;
            int elementCount = 1;

            for (int i = 0; i < dimensions.Length; i++)
                elementCount *= dimensions[i];

            if (elementCount != ModelInputSize * ModelInputSize * 3)
            {
                throw new InvalidDataException(
                    "The hand-landmark model input must contain 224 x 224 x 3 values.");
            }
        }

        private static bool TryCreateCropTransform(
            Size frameSize,
            PointF wrist,
            PointF elbow,
            float cropScale,
            out CropTransform transform)
        {
            transform = null;

            float forearmX = wrist.X - elbow.X;
            float forearmY = wrist.Y - elbow.Y;
            float forearmLength = (float)Math.Sqrt(
                forearmX * forearmX +
                forearmY * forearmY);

            if (forearmLength < 20f)
                return false;

            float forwardX = forearmX / forearmLength;
            float forwardY = forearmY / forearmLength;
            float rightX = -forwardY;
            float rightY = forwardX;
            float downX = -forwardX;
            float downY = -forwardY;

            float minimumSide = Math.Max(
                80f,
                Math.Min(frameSize.Width, frameSize.Height) * 0.055f);
            float maximumSide =
                Math.Min(frameSize.Width, frameSize.Height) * 0.70f;
            float safeCropScale = Math.Max(
                1f,
                Math.Min(1.5f, cropScale));
            float sideLength = Math.Max(
                minimumSide,
                Math.Min(
                    maximumSide,
                    forearmLength *
                    CropSizeFromForearm *
                    safeCropScale));

            PointF center = new PointF(
                wrist.X + forwardX * forearmLength * CropCenterFromWrist,
                wrist.Y + forwardY * forearmLength * CropCenterFromWrist);

            transform = new CropTransform(
                center,
                new PointF(rightX, rightY),
                new PointF(downX, downY),
                sideLength);

            return true;
        }

        private unsafe void FillInputTensor(Bitmap frame, CropTransform transform)
        {
            Rectangle rectangle = new Rectangle(0, 0, frame.Width, frame.Height);
            PixelFormat pixelFormat = frame.PixelFormat;
            int bytesPerPixel = Image.GetPixelFormatSize(pixelFormat) / 8;

            if (bytesPerPixel < 3)
                throw new InvalidOperationException("The camera frame must be an RGB bitmap.");

            BitmapData data = frame.LockBits(
                rectangle,
                ImageLockMode.ReadOnly,
                pixelFormat);

            try
            {
                byte* baseAddress = (byte*)data.Scan0.ToPointer();
                int destinationIndex = 0;
                float modelScale = transform.SideLength / ModelInputSize;

                for (int modelY = 0; modelY < ModelInputSize; modelY++)
                {
                    float offsetY =
                        (modelY + 0.5f - ModelInputSize / 2f) * modelScale;

                    for (int modelX = 0; modelX < ModelInputSize; modelX++)
                    {
                        float offsetX =
                            (modelX + 0.5f - ModelInputSize / 2f) * modelScale;

                        float sourceX =
                            transform.Center.X +
                            transform.Right.X * offsetX +
                            transform.Down.X * offsetY;
                        float sourceY =
                            transform.Center.Y +
                            transform.Right.Y * offsetX +
                            transform.Down.Y * offsetY;

                        float red;
                        float green;
                        float blue;

                        SampleBilinear(
                            baseAddress,
                            data.Stride,
                            bytesPerPixel,
                            frame.Width,
                            frame.Height,
                            sourceX,
                            sourceY,
                            out red,
                            out green,
                            out blue);

                        inputBuffer[destinationIndex++] = red;
                        inputBuffer[destinationIndex++] = green;
                        inputBuffer[destinationIndex++] = blue;
                    }
                }
            }
            finally
            {
                frame.UnlockBits(data);
            }
        }

        private static unsafe void SampleBilinear(
            byte* baseAddress,
            int stride,
            int bytesPerPixel,
            int width,
            int height,
            float x,
            float y,
            out float red,
            out float green,
            out float blue)
        {
            if (x < 0f || y < 0f || x >= width - 1f || y >= height - 1f)
            {
                red = 0f;
                green = 0f;
                blue = 0f;
                return;
            }

            int x0 = (int)x;
            int y0 = (int)y;
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            float xWeight = x - x0;
            float yWeight = y - y0;

            byte* p00 = baseAddress + y0 * stride + x0 * bytesPerPixel;
            byte* p10 = baseAddress + y0 * stride + x1 * bytesPerPixel;
            byte* p01 = baseAddress + y1 * stride + x0 * bytesPerPixel;
            byte* p11 = baseAddress + y1 * stride + x1 * bytesPerPixel;

            blue = InterpolateChannel(
                p00[0],
                p10[0],
                p01[0],
                p11[0],
                xWeight,
                yWeight);
            green = InterpolateChannel(
                p00[1],
                p10[1],
                p01[1],
                p11[1],
                xWeight,
                yWeight);
            red = InterpolateChannel(
                p00[2],
                p10[2],
                p01[2],
                p11[2],
                xWeight,
                yWeight);
        }

        private static float InterpolateChannel(
            byte topLeft,
            byte topRight,
            byte bottomLeft,
            byte bottomRight,
            float xWeight,
            float yWeight)
        {
            float top = topLeft + (topRight - topLeft) * xWeight;
            float bottom = bottomLeft + (bottomRight - bottomLeft) * xWeight;
            return (top + (bottom - top) * yWeight) / 255f;
        }

        private bool TryReadOutputs(
            out float[] screenLandmarks,
            out float handPresence)
        {
            screenLandmarks = null;
            handPresence = 0f;

            List<float[]> landmarkCandidates = new List<float[]>();
            List<KeyValuePair<string, float>> scalarCandidates =
                new List<KeyValuePair<string, float>>();

            for (int i = 0; i < outputTensors.Length; i++)
            {
                Tensor tensor = outputTensors[i];

                if (tensor.Type != DataType.Float32)
                    continue;

                float[] values = tensor.Data as float[];
                if (values == null)
                    continue;

                if (values.Length == LandmarkValueCount)
                {
                    if (string.Equals(
                        tensor.Name,
                        "Identity",
                        StringComparison.OrdinalIgnoreCase))
                    {
                        screenLandmarks = values;
                    }

                    landmarkCandidates.Add(values);
                }
                else if (values.Length == 1)
                {
                    scalarCandidates.Add(
                        new KeyValuePair<string, float>(
                            tensor.Name ?? string.Empty,
                            values[0]));
                }
            }

            if (screenLandmarks == null)
                screenLandmarks = SelectScreenLandmarks(landmarkCandidates);

            if (screenLandmarks == null)
                return false;

            bool foundPresence = false;

            for (int i = 0; i < scalarCandidates.Count; i++)
            {
                string name = scalarCandidates[i].Key;

                if (string.Equals(
                    name,
                    "Identity_1",
                    StringComparison.OrdinalIgnoreCase) ||
                    name.IndexOf("presence", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    name.IndexOf("handflag", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    handPresence = NormalizeProbability(scalarCandidates[i].Value);
                    foundPresence = true;
                    break;
                }
            }

            if (!foundPresence && scalarCandidates.Count > 0)
            {
                handPresence =
                    NormalizeProbability(scalarCandidates[0].Value);
                foundPresence = true;
            }

            return foundPresence;
        }

        private static float[] SelectScreenLandmarks(
            IList<float[]> candidates)
        {
            float[] best = null;
            float bestMagnitude = float.MinValue;

            for (int i = 0; i < candidates.Count; i++)
            {
                float magnitude = 0f;
                float[] values = candidates[i];

                for (int j = 0; j < values.Length; j += 3)
                {
                    magnitude = Math.Max(magnitude, Math.Abs(values[j]));
                    magnitude = Math.Max(magnitude, Math.Abs(values[j + 1]));
                }

                if (magnitude > bestMagnitude)
                {
                    bestMagnitude = magnitude;
                    best = values;
                }
            }

            return best;
        }

        private static float NormalizeProbability(float value)
        {
            if (value >= 0f && value <= 1f)
                return value;

            return 1f / (1f + (float)Math.Exp(-value));
        }

        private static PointF ReadLandmark(float[] values, int index)
        {
            int offset = index * 3;
            return new PointF(values[offset], values[offset + 1]);
        }

        private static float Distance(PointF first, PointF second)
        {
            float dx = second.X - first.X;
            float dy = second.Y - first.Y;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }

        public void Dispose()
        {
            lock (syncRoot)
            {
                if (disposed)
                    return;

                disposed = true;

                if (interpreter != null)
                {
                    interpreter.Dispose();
                    interpreter = null;
                }

                if (model != null)
                {
                    model.Dispose();
                    model = null;
                }

                inputTensor = null;
                outputTensors = null;
            }
        }

        private sealed class CropTransform
        {
            public CropTransform(
                PointF center,
                PointF right,
                PointF down,
                float sideLength)
            {
                Center = center;
                Right = right;
                Down = down;
                SideLength = sideLength;
            }

            public PointF Center { get; private set; }
            public PointF Right { get; private set; }
            public PointF Down { get; private set; }
            public float SideLength { get; private set; }

            public RectangleF AxisAlignedBounds
            {
                get
                {
                    float half = SideLength / 2f;
                    return new RectangleF(
                        Center.X - half,
                        Center.Y - half,
                        SideLength,
                        SideLength);
                }
            }

            public PointF MapToFrame(PointF modelPoint)
            {
                float offsetX =
                    (modelPoint.X / ModelInputSize - 0.5f) * SideLength;
                float offsetY =
                    (modelPoint.Y / ModelInputSize - 0.5f) * SideLength;

                return new PointF(
                    Center.X + Right.X * offsetX + Down.X * offsetY,
                    Center.Y + Right.Y * offsetX + Down.Y * offsetY);
            }
        }
    }
}
