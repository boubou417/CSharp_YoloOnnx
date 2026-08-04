using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SpinnakerNET;
using SpinnakerNET.GenApi;

namespace CSharp_YoloOnnx
{
    public partial class Form1 : Form
    {
        IManagedCamera cam;
        bool connected = false;
        bool streaming = false;
        bool grabImage = false;
        bool threadComplete = false;
        // ===== Gesture Timer =====
        DateTime handRaisedStart = DateTime.MinValue;
        const int RaiseHandDelayMs = 800;

        // ===== 揮手偵測 =====
        Queue<float> rightHandHistory = new Queue<float>();
        bool isWaving = false;

        // ===== 即時軌跡(畫面顯示) =====
        List<PointF> handTrail = new List<PointF>();

        GameState gameState = GameState.Idle;
        List<PointF> drawingPoints = new List<PointF>();
        List<DateTime> drawingPointTimes = new List<DateTime>();
        List<int> drawingStrokeStartIndices = new List<int>();

        const int MinimumDrawingPoints = 12;
        const int FinishedDisplayMs = 2500;
        const int MagicAnimationDisplayMs = 1800;
        const int FingertipMissingBreakMs = 500;
        const int FingertipMarkerVisibleMs = 400;
        const int FingertipInferenceIntervalMs = 67;
        const int IdleHandInferenceIntervalMs = 120;
        const int OpenPalmStartDelayMs = 600;
        const int StartGestureMissingGraceMs = 300;
        const int HandTrackingFallbackMs = 300;
        const int PinchPreconfirmMs = 180;
        const int PinchFinishDelayMs = 600;
        const int PinchTailTrimMs = 250;
        const float MinimumPointDistance = 2f;
        const float PointSmoothingFactor = 0.60f;
        const float FingertipSmoothingFactor = 0.70f;
        const float PinchStartRatio = 0.32f;
        const float PinchReleaseRatio = 0.42f;
        const float OpenPalmStartScore = 0.75f;
        const float OpenPalmReleaseScore = 0.50f;
        const float MinimumPoseHandKeypointScore = 0.35f;
        const float MinimumIdleHandPresence = 0.35f;
        const float MinimumDrawingHandPresence = 0.45f;
        const int YellowStartHoldMs = 600;
        const int YellowFinishHoldMs = 900;
        const int YellowMissingBreakMs = 300;
        const int YellowReacquireConfirmMs = 200;
        const int YellowTrackingLossGraceMs = 120;
        const int YellowMaximumInterpolatedPoints = 16;
        const float YellowHoldRadius = 30f;
        const float YellowReacquireRadius = 36f;
        const float YellowBaseFrameJump = 80f;
        const float YellowMaximumSpeedPixelsPerSecond = 7000f;
        const float YellowMaximumAdaptiveJump = 480f;
        const float YellowJitterDeadZone = 4f;
        const float YellowSlowPositionSmoothing = 0.55f;
        const float YellowFastPositionSmoothing = 0.85f;
        const float YellowFastMotionDistance = 60f;
        const float YellowInterpolationSpacing = 16f;
        const float YellowStartMoveDistance = 40f;

        DateTime drawingFinishedAt = DateTime.MinValue;
        DateTime fingertipMissingSince = DateTime.MinValue;
        DateTime lastFingertipSeenAt = DateTime.MinValue;
        DateTime lastFingertipInferenceAt = DateTime.MinValue;
        DateTime lastIdleHandInferenceAt = DateTime.MinValue;
        DateTime leftStartGestureAt = DateTime.MinValue;
        DateTime rightStartGestureAt = DateTime.MinValue;
        DateTime leftStartGestureLastSeenAt = DateTime.MinValue;
        DateTime rightStartGestureLastSeenAt = DateTime.MinValue;
        DateTime pinchCandidateStartedAt = DateTime.MinValue;
        DateTime pinchStartedAt = DateTime.MinValue;
        string drawingStatusText =
            "左右手皆可｜張開手掌 0.6 秒開始";
        string templateImagePath = string.Empty;
        double? lastDrawingScore;
        Button btnSelectTemplate;
        Label lblSimilarityScore;
        Label lblAirDrawStatus;
        FingertipTracker fingertipTracker;
        readonly YellowTipTracker yellowTipTracker =
            new YellowTipTracker();
        readonly bool yellowTipMode = true;
        DateTime yellowHoldStartedAt = DateTime.MinValue;
        DateTime yellowMissingSince = DateTime.MinValue;
        PointF? yellowHoldAnchor;
        PointF? yellowDrawingStartPoint;
        RectangleF displayedYellowTipBounds;
        bool yellowMovedAfterStart;
        bool yellowTrackingConfirmed;
        DateTime yellowCandidateStartedAt = DateTime.MinValue;
        DateTime yellowRawMissingSince = DateTime.MinValue;
        DateTime lastYellowAcceptedAt = DateTime.MinValue;
        PointF? yellowCandidateAnchor;
        PointF? filteredYellowTip;
        PointF? lastFingertipPoint;
        PointF? displayedFingertipPoint;
        PointF? displayedThumbPoint;
        float displayedFingertipPresence;
        float latestPinchRatio = float.MaxValue;
        float latestOpenPalmScore;
        bool drawingStrokeStartPending = true;
        bool pinchInProgress;
        bool waitingForOpenPalmReleaseAfterStart;
        DrawingHand activeDrawingHand = DrawingHand.None;
        DrawingHand displayedDrawingHand = DrawingHand.None;
        HandTrackingAnchor leftHandTracking =
            new HandTrackingAnchor();
        HandTrackingAnchor rightHandTracking =
            new HandTrackingAnchor();
        Bitmap pendingDisplayImage;
        int displayUpdateScheduled;
        Bitmap pendingPoseImage;
        readonly AutoResetEvent poseFrameReady =
            new AutoResetEvent(false);
        volatile PoseSnapshot latestPoseSnapshot =
            PoseSnapshot.Empty;
        Thread acquisitionThread;
        Thread poseInferenceThread;
        bool poseThreadComplete;
        int poseFrameSlotReserved;
        int handInferenceBusy;
        Task handInferenceTask;
        volatile HandInferenceSnapshot latestHandSnapshot;
        DateTime lastConsumedLeftHandAt = DateTime.MinValue;
        DateTime lastConsumedRightHandAt = DateTime.MinValue;
        DrawingHand nextIdleHandRequest = DrawingHand.Left;
        readonly Stopwatch cameraFpsStopwatch =
            Stopwatch.StartNew();
        readonly Stopwatch displayFpsStopwatch =
            Stopwatch.StartNew();
        int cameraFrameCount;
        int displayFrameCount;
        double cameraFramesPerSecond;
        double displayFramesPerSecond;
        double poseFramesPerSecond;
        double latestHandInferenceMilliseconds;
        string latestHandInferenceResult = "WAIT";
        Image magicAnimationImage;
        MemoryStream magicAnimationStream;
        DateTime magicAnimationStartedAt = DateTime.MinValue;

        InferenceSession yoloSession;
        string yoloExecutionProvider = "CPU";

        // The bundled YOLO pose model has a fixed 640 x 640 input.
        // Keep these dimensions aligned with the ONNX input metadata.
        const int yoloImgWidth = 640;
        const int yoloImgHeight = 640;

        float _ratio;
        int newW;
        int newH;
        int _padX;
        int _padY;

        // 🔥 Pose Keypoint
        public class Keypoint
        {
            public float X;
            public float Y;
            public float Score;
        }

        public class Detection
        {
            public float X;
            public float Y;
            public float W;
            public float H;
            public float Score;
            public List<Keypoint> Keypoints = new List<Keypoint>();
        }

        // ===============================
        // 遊戲狀態
        // ===============================
        public enum GameState
        {
            Idle,        // 等待玩家
            Countdown,   // 倒數
            Drawing,     // 畫圖中
            Finished,    // 已完成
            Scoring      // 評分中
        }

        public enum DrawingHand
        {
            None,
            Left,
            Right
        }

        private sealed class PoseSnapshot
        {
            public static readonly PoseSnapshot Empty =
                new PoseSnapshot(
                    new List<Detection>(),
                    null,
                    0d);

            public readonly List<Detection> Boxes;
            public readonly Detection Main;
            public readonly double InferenceMilliseconds;

            public PoseSnapshot(
                List<Detection> boxes,
                Detection main,
                double inferenceMilliseconds)
            {
                Boxes = boxes ?? new List<Detection>();
                Main = main;
                InferenceMilliseconds = inferenceMilliseconds;
            }
        }

        private sealed class HandInferenceSnapshot
        {
            public readonly DrawingHand Hand;
            public readonly FingertipResult Result;
            public readonly bool Detected;
            public readonly DateTime CompletedAt;
            public readonly PointF ForearmOffset;
            public readonly double Milliseconds;
            public readonly Exception Error;

            public HandInferenceSnapshot(
                DrawingHand hand,
                FingertipResult result,
                bool detected,
                DateTime completedAt,
                PointF forearmOffset,
                double milliseconds,
                Exception error)
            {
                Hand = hand;
                Result = result;
                Detected = detected;
                CompletedAt = completedAt;
                ForearmOffset = forearmOffset;
                Milliseconds = milliseconds;
                Error = error;
            }
        }

        private sealed class HandTrackingAnchor
        {
            public PointF Wrist;
            public PointF Elbow;
            public DateTime UpdatedAt = DateTime.MinValue;

            public bool IsRecent(DateTime now)
            {
                return UpdatedAt != DateTime.MinValue &&
                    (now - UpdatedAt).TotalMilliseconds <=
                        HandTrackingFallbackMs;
            }
        }

        public Form1()
        {
            InitializeComponent();

            Text = "CSharp YOLO ONNX V1.5.5 Connected Red-Yellow Pointer";
            panelToolBar.Dock = DockStyle.Top;
            panelToolBar.Height = 40;
            panelStatusBar.Dock = DockStyle.Bottom;
            panelStatusBar.Height = 64;
            panelImage.Dock = DockStyle.Fill;
            pBox.Dock = DockStyle.Fill;
            pBox.SizeMode = PictureBoxSizeMode.Zoom;
            InitializeAirDrawControls();
            TryLoadDefaultTemplate();
            if (!yellowTipMode)
                InitializeFingertipTracker();
            InitializeMagicAnimation();
            FormClosed += Form1_FormClosed;

            ManagedSystem system = new ManagedSystem();
            IList<IManagedCamera> camList = system.GetCameras();

            if (camList.Count == 0)
            {
                MessageBox.Show("No Cameras!!");
                Environment.Exit(Environment.ExitCode);
            }
            else
            {
                cam = camList[0];
            }

            string modelPath = "yolov8n-pose.onnx";
            InitializeYoloSession(modelPath);
            Text =
                "CSharp YOLO ONNX V1.5.5 Connected Red-Yellow Pointer | " +
                yoloExecutionProvider;
        }

        private void InitializeYoloSession(
            string modelPath)
        {
            try
            {
                using (SessionOptions cudaOptions =
                    new SessionOptions())
                {
                    ConfigureCommonSessionOptions(
                        cudaOptions,
                        false);
                    cudaOptions.AppendExecutionProvider_CUDA(0);

                    yoloSession =
                        new InferenceSession(
                            modelPath,
                            cudaOptions);
                }

                yoloExecutionProvider = "CUDA GPU";
                Debug.WriteLine(
                    "YOLO execution provider = CUDA GPU");
                return;
            }
            catch (Exception cudaException)
            {
                if (yoloSession != null)
                {
                    yoloSession.Dispose();
                    yoloSession = null;
                }

                Debug.WriteLine(
                    "CUDA is unavailable; falling back to CPU. " +
                    cudaException);
            }

            using (SessionOptions cpuOptions =
                new SessionOptions())
            {
                ConfigureCommonSessionOptions(
                    cpuOptions,
                    true);

                yoloSession =
                    new InferenceSession(
                        modelPath,
                        cpuOptions);
            }

            yoloExecutionProvider = "CPU fallback";
            Debug.WriteLine(
                "YOLO execution provider = CPU fallback");
        }

        private static void ConfigureCommonSessionOptions(
            SessionOptions sessionOptions,
            bool configureCpuThreads)
        {
            sessionOptions.GraphOptimizationLevel =
                GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.ExecutionMode =
                ExecutionMode.ORT_SEQUENTIAL;

            if (!configureCpuThreads)
                return;

            sessionOptions.IntraOpNumThreads =
                Math.Max(
                    1,
                    Environment.ProcessorCount - 1);
            sessionOptions.InterOpNumThreads = 1;
        }

        private void InitializeFingertipTracker()
        {
            string modelPath = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "Models",
                "hand_landmark_full.tflite");

            try
            {
                fingertipTracker = new FingertipTracker(modelPath);
                Debug.WriteLine("Fingertip model = " + modelPath);
            }
            catch (Exception ex)
            {
                fingertipTracker = null;
                drawingStatusText =
                    "指尖模型無法載入｜請確認 Models/hand_landmark_full.tflite";
                Debug.WriteLine("Fingertip tracker initialization error: " + ex);
            }
        }

        private void Form1_FormClosed(object sender, FormClosedEventArgs e)
        {
            grabImage = false;
            poseFrameReady.Set();

            if (acquisitionThread != null)
                acquisitionThread.Join(1500);

            if (poseInferenceThread != null)
                poseInferenceThread.Join(1500);

            WaitForHandInference(1500);
            DisposePendingDisplayImage();
            DisposePendingPoseImage();

            if (yoloSession != null)
            {
                yoloSession.Dispose();
                yoloSession = null;
            }
            DisposeMagicAnimation();

            if (fingertipTracker != null)
            {
                fingertipTracker.Dispose();
                fingertipTracker = null;
            }
        }

        private void InitializeMagicAnimation()
        {
            string animationPath = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "Assets",
                "magic-cast-test.gif");

            try
            {
                byte[] animationBytes = File.ReadAllBytes(animationPath);
                magicAnimationStream =
                    new MemoryStream(animationBytes, false);
                magicAnimationImage =
                    Image.FromStream(magicAnimationStream);
                Debug.WriteLine("Magic animation = " + animationPath);
            }
            catch (Exception ex)
            {
                DisposeMagicAnimation();
                Debug.WriteLine(
                    "Magic animation initialization error: " + ex);
            }
        }

        private void DisposeMagicAnimation()
        {
            if (magicAnimationImage != null)
            {
                ImageAnimator.StopAnimate(
                    magicAnimationImage,
                    MagicAnimationFrameChanged);
                magicAnimationImage.Dispose();
                magicAnimationImage = null;
            }

            if (magicAnimationStream != null)
            {
                magicAnimationStream.Dispose();
                magicAnimationStream = null;
            }

            magicAnimationStartedAt = DateTime.MinValue;
        }

        private bool StartMagicAnimation()
        {
            if (magicAnimationImage == null ||
                !ImageAnimator.CanAnimate(magicAnimationImage))
            {
                return false;
            }

            ImageAnimator.StopAnimate(
                magicAnimationImage,
                MagicAnimationFrameChanged);

            Guid[] dimensions =
                magicAnimationImage.FrameDimensionsList;

            if (dimensions.Length > 0)
            {
                magicAnimationImage.SelectActiveFrame(
                    new FrameDimension(dimensions[0]),
                    0);
            }

            magicAnimationStartedAt = DateTime.Now;
            ImageAnimator.Animate(
                magicAnimationImage,
                MagicAnimationFrameChanged);
            return true;
        }

        private void MagicAnimationFrameChanged(
            object sender,
            EventArgs e)
        {
            // Live camera frames trigger repainting; no extra UI refresh is
            // scheduled here so the GIF does not add rendering work.
        }

        private void DrawMagicAnimation(
            Graphics graphics,
            int imageWidth,
            int imageHeight)
        {
            if (magicAnimationImage == null ||
                magicAnimationStartedAt == DateTime.MinValue)
            {
                return;
            }

            if ((DateTime.Now - magicAnimationStartedAt)
                    .TotalMilliseconds >= MagicAnimationDisplayMs)
            {
                ImageAnimator.StopAnimate(
                    magicAnimationImage,
                    MagicAnimationFrameChanged);
                magicAnimationStartedAt = DateTime.MinValue;
                return;
            }

            ImageAnimator.UpdateFrames(magicAnimationImage);

            int animationSize = (int)Math.Min(
                Math.Min(imageWidth, imageHeight) * 0.72f,
                640f);
            int left = (imageWidth - animationSize) / 2;
            int top = (imageHeight - animationSize) / 2;

            graphics.DrawImage(
                magicAnimationImage,
                new Rectangle(
                    left,
                    top,
                    animationSize,
                    animationSize));
        }

        private void InitializeAirDrawControls()
        {
            btnSelectTemplate = new Button
            {
                Name = "btnSelectTemplate",
                Text = "選擇比對圖",
                Width = 110,
                Height = 26,
                Left = btnGrab.Right + 20,
                Top = 7
            };

            btnSelectTemplate.Click += btnSelectTemplate_Click;
            panelToolBar.Controls.Add(btnSelectTemplate);

            lblSimilarityScore = new Label
            {
                Dock = DockStyle.Left,
                Width = 300,
                TextAlign = ContentAlignment.MiddleCenter,
                BorderStyle = BorderStyle.FixedSingle,
                Font = new Font(
                    "Microsoft JhengHei UI",
                    16f,
                    FontStyle.Bold),
                ForeColor = Color.DimGray,
                BackColor = Color.White,
                Text = "形狀相似度：--"
            };

            lblAirDrawStatus = new Label
            {
                Dock = DockStyle.Fill,
                AutoEllipsis = true,
                TextAlign = ContentAlignment.MiddleLeft,
                Padding = new Padding(12, 0, 6, 0),
                ForeColor = Color.Black,
                BackColor = SystemColors.Control,
                Text = drawingStatusText
            };

            panelStatusBar.Controls.Add(lblAirDrawStatus);
            panelStatusBar.Controls.Add(lblSimilarityScore);
            lblSimilarityScore.BringToFront();
        }

        private void TryLoadDefaultTemplate()
        {
            string defaultTemplatePath = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "Templates",
                "template.png");

            if (!File.Exists(defaultTemplatePath))
                return;

            templateImagePath = defaultTemplatePath;
            drawingStatusText =
                "比對圖：" +
                Path.GetFileName(templateImagePath) +
                "｜黃色筆尖停留開始";
        }

        private void btnSelectTemplate_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Title = "選擇要比對的字卡或圖卡";
                dialog.Filter =
                    "圖片檔案|*.png;*.jpg;*.jpeg;*.bmp|" +
                    "所有檔案|*.*";

                if (dialog.ShowDialog(this) != DialogResult.OK)
                    return;

                templateImagePath = dialog.FileName;
                drawingStatusText =
                    "比對圖：" +
                    Path.GetFileName(templateImagePath) +
                    "｜左右手張掌開始";
            }
        }

        private void StreamBufferMode(INodeMap nodeMap)
        {
            IEnum iStreamBufferHandlingMode = nodeMap.GetNode<IEnum>("StreamBufferHandlingMode");
            iStreamBufferHandlingMode.Value = "NewestOnly";
        }

        private void btnConnect_Click(object sender, EventArgs e)
        {
            if (!connected)
            {
                cam.Init();
                INodeMap nodeMap = cam.GetNodeMap();
                parameterInit(nodeMap);

                INodeMap sNodeMap = cam.GetTLStreamNodeMap();
                StreamBufferMode(sNodeMap);

                connected = true;
                btnConnect.Text = "Disconnect";
            }
            else
            {
                cam.DeInit();
                connected = false;
                btnConnect.Text = "Connect";
            }
        }

        private void parameterInit(INodeMap nodeMap)
        {
            IInteger iWidth = nodeMap.GetNode<IInteger>("Width");
            IInteger iHeight = nodeMap.GetNode<IInteger>("Height");

            _ratio = Math.Min((float)yoloImgWidth / iWidth.Value, (float)yoloImgHeight / iHeight.Value);

            newW = (int)(iWidth.Value * _ratio);
            newH = (int)(iHeight.Value * _ratio);

            _padX = (yoloImgWidth - newW) / 2;
            _padY = (yoloImgHeight - newH) / 2;
        }

        private void btnGrab_Click(object sender, EventArgs e)
        {
            if (!streaming)
            {
                cam.BeginAcquisition();
                grabImage = true;
                threadComplete = false;
                poseThreadComplete = false;
                latestPoseSnapshot = PoseSnapshot.Empty;
                latestHandSnapshot = null;
                lastConsumedLeftHandAt = DateTime.MinValue;
                lastConsumedRightHandAt = DateTime.MinValue;
                nextIdleHandRequest = DrawingHand.Left;
                Interlocked.Exchange(
                    ref handInferenceBusy,
                    0);
                DisposePendingPoseImage();
                ResetPerformanceDiagnostics();
                ResetYellowTracking();

                poseInferenceThread =
                    new Thread(ThreadPoseInference)
                    {
                        IsBackground = true,
                        Priority = ThreadPriority.Normal,
                        Name = "YOLO pose inference"
                    };
                acquisitionThread =
                    new Thread(ThreadGetImages)
                    {
                        IsBackground = true,
                        Priority = ThreadPriority.AboveNormal,
                        Name = "Camera acquisition and display"
                    };

                poseInferenceThread.Start();
                acquisitionThread.Start();

                streaming = true;
                btnGrab.Text = "Stop";
            }
            else
            {
                grabImage = false;
                poseFrameReady.Set();

                while (!threadComplete || !poseThreadComplete)
                {
                    Application.DoEvents();
                    Thread.Sleep(10);
                }

                WaitForHandInference(1500);
                cam.EndAcquisition();
                DisposePendingPoseImage();

                streaming = false;
                threadComplete = false;
                poseThreadComplete = false;
                acquisitionThread = null;
                poseInferenceThread = null;
                btnGrab.Text = "Grab";
            }
        }

        public delegate void InvokeDelegate(Bitmap bmp);

        private void ThreadGetImages()
        {
            IManagedImageProcessor processor =
                new ManagedImageProcessor();
            IManagedImage convertedImage =
                new ManagedImage();

            try
            {
                while (grabImage)
                {
                    IManagedImage rawImage =
                        cam.GetNextImage();

                    try
                    {
                        processor.Convert(
                            rawImage,
                            convertedImage,
                            PixelFormatEnums.BGR8);
                    }
                    finally
                    {
                        rawImage.Release();
                    }

                    Bitmap displayImage =
                        CopyManagedImageToBitmap(
                            convertedImage);

                    QueueLatestPoseFrame(displayImage);

                    PoseSnapshot snapshot =
                        latestPoseSnapshot ??
                        PoseSnapshot.Empty;

                    RenderDisplayImage(
                        displayImage,
                        snapshot.Boxes,
                        snapshot.Main,
                        snapshot.InferenceMilliseconds);
                    QueueDisplayImage(displayImage);
                    RecordCameraFrame();
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(
                    "Camera/display thread error: " + ex);
            }
            finally
            {
                threadComplete = true;
                poseFrameReady.Set();
            }
        }

        private void ThreadPoseInference()
        {
            DenseTensor<float> tensor =
                new DenseTensor<float>(
                    new[]
                    {
                        1,
                        3,
                        yoloImgHeight,
                        yoloImgWidth
                    });

            try
            {
                while (grabImage ||
                    Interlocked.CompareExchange(
                        ref pendingPoseImage,
                        null,
                        null) != null)
                {
                    poseFrameReady.WaitOne(100);

                    Bitmap inferenceImage =
                        Interlocked.Exchange(
                            ref pendingPoseImage,
                            null);

                    if (inferenceImage == null)
                        continue;

                    try
                    {
                        Stopwatch stopwatch =
                            Stopwatch.StartNew();

                        CreateTensorFromBitmap(
                            inferenceImage,
                            ref tensor);

                        List<Detection> finalBoxes;

                        using (var output =
                            yoloSession.Run(new[]
                            {
                                NamedOnnxValue.CreateFromTensor(
                                    "images",
                                    tensor)
                            }))
                        {
                            Tensor<float> resultTensor =
                                output.First()
                                    .AsTensor<float>();
                            finalBoxes =
                                PostProcess(resultTensor);
                        }

                        Detection main =
                            finalBoxes
                                .OrderByDescending(
                                    detection =>
                                        detection.W *
                                        detection.H)
                                .FirstOrDefault();

                        stopwatch.Stop();
                        poseFramesPerSecond =
                            stopwatch.Elapsed.TotalMilliseconds >
                                0d
                                ? 1000d /
                                  stopwatch.Elapsed.TotalMilliseconds
                                : 0d;

                        latestPoseSnapshot =
                            new PoseSnapshot(
                                finalBoxes,
                                main,
                                stopwatch.Elapsed
                                    .TotalMilliseconds);
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine(
                            "YOLO inference thread error: " +
                            ex);
                    }
                    finally
                    {
                        inferenceImage.Dispose();
                        Interlocked.Exchange(
                            ref poseFrameSlotReserved,
                            0);
                    }
                }
            }
            finally
            {
                poseThreadComplete = true;
            }
        }

        private void QueueLatestPoseFrame(
            Bitmap displayImage)
        {
            if (displayImage == null || !grabImage)
                return;

            // Reserve only one frame for YOLO. While inference is running,
            // camera/display frames continue without cloning large Bitmaps.
            if (Interlocked.CompareExchange(
                ref poseFrameSlotReserved,
                1,
                0) != 0)
            {
                return;
            }

            Bitmap inferenceCopy = null;

            try
            {
                inferenceCopy = displayImage.Clone(
                    new Rectangle(
                        0,
                        0,
                        displayImage.Width,
                        displayImage.Height),
                    PixelFormat.Format24bppRgb);

                Bitmap replaced =
                    Interlocked.Exchange(
                        ref pendingPoseImage,
                        inferenceCopy);
                inferenceCopy = null;
                replaced?.Dispose();
                poseFrameReady.Set();
            }
            catch
            {
                Interlocked.Exchange(
                    ref poseFrameSlotReserved,
                    0);
                throw;
            }
            finally
            {
                inferenceCopy?.Dispose();
            }
        }

        private void DisposePendingPoseImage()
        {
            Bitmap pending =
                Interlocked.Exchange(
                    ref pendingPoseImage,
                    null);
            pending?.Dispose();
            Interlocked.Exchange(
                ref poseFrameSlotReserved,
                0);
        }

        private void StartDrawing(
            DrawingHand hand,
            FingertipResult startResult)
        {
            drawingPoints.Clear();
            drawingPointTimes.Clear();
            drawingStrokeStartIndices.Clear();
            handTrail.Clear();
            rightHandHistory.Clear();
            isWaving = false;
            handRaisedStart = DateTime.MinValue;
            drawingFinishedAt = DateTime.MinValue;
            fingertipMissingSince = DateTime.MinValue;
            lastFingertipSeenAt = DateTime.MinValue;
            lastFingertipInferenceAt = DateTime.MinValue;
            lastFingertipPoint = null;
            displayedFingertipPoint = null;
            displayedThumbPoint = null;
            displayedFingertipPresence = 0f;
            ResetPinchGesture(false);
            ResetStartGestures();
            drawingStrokeStartPending = true;
            lastDrawingScore = null;
            activeDrawingHand = hand;
            displayedDrawingHand = hand;
            waitingForOpenPalmReleaseAfterStart = true;

            if (startResult != null)
            {
                displayedFingertipPoint =
                    startResult.IndexTip;
                displayedThumbPoint =
                    startResult.ThumbTip;
                displayedFingertipPresence =
                    startResult.HandPresence;
                latestPinchRatio =
                    startResult.PinchRatio;
                latestOpenPalmScore =
                    startResult.OpenPalmScore;
                lastFingertipSeenAt = DateTime.Now;
            }

            gameState = GameState.Drawing;
            drawingStatusText =
                GetHandDisplayName(hand) +
                "開始成功｜請收起其他手指並用食指畫圖";

            Debug.WriteLine("===== Start Drawing =====");
        }

        private void StopDrawing(
            DateTime gestureStartedAt,
            int tailTrimMs)
        {
            if (gameState != GameState.Drawing)
                return;

            TrimGestureTail(gestureStartedAt, tailTrimMs);
            ResetPinchGesture(false);
            waitingForOpenPalmReleaseAfterStart = false;
            rightHandHistory.Clear();
            isWaving = false;
            handRaisedStart = DateTime.MinValue;
            displayedFingertipPoint = null;
            displayedThumbPoint = null;
            displayedDrawingHand = DrawingHand.None;
            activeDrawingHand = DrawingHand.None;

            Debug.WriteLine("===== Finish Drawing =====");
            Debug.WriteLine($"Trajectory Points = {drawingPoints.Count}");
            bool drawingSaved = false;

            if (drawingPoints.Count < MinimumDrawingPoints)
            {
                drawingStatusText =
                    "軌跡太短，未儲存｜請重新張掌開始";
                drawingFinishedAt = DateTime.Now;
                gameState = GameState.Finished;
                return;
            }

            try
            {
                gameState = GameState.Scoring;

                string outputDirectory = Path.Combine(
                    AppDomain.CurrentDomain.BaseDirectory,
                    "Drawings");

                AirDrawSaveResult saveResult = AirDrawStorage.Save(
                    drawingPoints,
                    drawingStrokeStartIndices,
                    outputDirectory);
                drawingSaved = true;

                string selectedTemplate = templateImagePath;

                if (!string.IsNullOrWhiteSpace(selectedTemplate) &&
                    File.Exists(selectedTemplate))
                {
                    lastDrawingScore = AirDrawComparer.Compare(
                        saveResult.ImagePath,
                        selectedTemplate);

                    AirDrawStorage.WriteMetadata(
                        saveResult,
                        selectedTemplate,
                        lastDrawingScore);

                    drawingStatusText =
                        "完成｜相似度 " +
                        lastDrawingScore.Value.ToString("0.0") +
                        " 分｜" +
                        Path.GetFileName(saveResult.ImagePath);
                }
                else
                {
                    drawingStatusText =
                        "完成並儲存｜尚未選擇比對圖｜" +
                        Path.GetFileName(saveResult.ImagePath);
                }

                Debug.WriteLine("Drawing image = " + saveResult.ImagePath);
                Debug.WriteLine("Drawing data = " + saveResult.JsonPath);
            }
            catch (Exception ex)
            {
                lastDrawingScore = null;
                drawingStatusText = "軌跡儲存或比對失敗，請查看 Debug 輸出";
                Debug.WriteLine("Air Draw error: " + ex);
            }

            if (drawingSaved && StartMagicAnimation())
                drawingStatusText += "｜魔法施放！";

            drawingFinishedAt = DateTime.Now;
            gameState = GameState.Finished;
        }

        private void AddDrawingPoint(PointF pt)
        {
            if (gameState != GameState.Drawing)
                return;

            PointF filteredPoint = pt;
            bool startsNewStroke =
                drawingStrokeStartPending ||
                drawingPoints.Count == 0;

            if (!startsNewStroke)
            {
                PointF previous = drawingPoints[drawingPoints.Count - 1];

                filteredPoint = new PointF(
                    previous.X + (pt.X - previous.X) * PointSmoothingFactor,
                    previous.Y + (pt.Y - previous.Y) * PointSmoothingFactor);

                float dx = filteredPoint.X - previous.X;
                float dy = filteredPoint.Y - previous.Y;

                if (dx * dx + dy * dy < MinimumPointDistance * MinimumPointDistance)
                    return;
            }

            if (startsNewStroke)
            {
                drawingStrokeStartIndices.Add(drawingPoints.Count);
                drawingStrokeStartPending = false;
            }

            drawingPoints.Add(filteredPoint);
            drawingPointTimes.Add(DateTime.Now);
            handTrail.Add(filteredPoint);
        }

        private void TrimGestureTail(
            DateTime gestureStartedAt,
            int tailTrimMs)
        {
            if (drawingPoints.Count == 0 ||
                drawingPointTimes.Count != drawingPoints.Count)
                return;

            DateTime detectedAt =
                gestureStartedAt == DateTime.MinValue
                    ? DateTime.Now
                    : gestureStartedAt;

            DateTime cutoff =
                detectedAt.AddMilliseconds(-tailTrimMs);
            int keepCount = drawingPointTimes.Count;

            while (keepCount > 0 && drawingPointTimes[keepCount - 1] >= cutoff)
                keepCount--;

            if (keepCount >= drawingPoints.Count)
                return;

            int removeCount = drawingPoints.Count - keepCount;
            drawingPoints.RemoveRange(keepCount, removeCount);
            drawingPointTimes.RemoveRange(keepCount, removeCount);

            if (handTrail.Count >= keepCount + removeCount)
                handTrail.RemoveRange(keepCount, removeCount);

            for (int i = drawingStrokeStartIndices.Count - 1; i >= 0; i--)
            {
                if (drawingStrokeStartIndices[i] >= keepCount)
                    drawingStrokeStartIndices.RemoveAt(i);
            }

            if (drawingPoints.Count > 0 &&
                (drawingStrokeStartIndices.Count == 0 ||
                 drawingStrokeStartIndices[0] != 0))
            {
                drawingStrokeStartIndices.Insert(0, 0);
            }
        }

        private void UpdateGame(Detection main, Bitmap frame)
        {
            if (yellowTipMode)
            {
                UpdateYellowTipGame(frame);
                return;
            }

            if (gameState == GameState.Finished)
            {
                if (drawingFinishedAt != DateTime.MinValue &&
                    (DateTime.Now - drawingFinishedAt).TotalMilliseconds >= FinishedDisplayMs)
                {
                    gameState = GameState.Idle;
                    handTrail.Clear();
                    drawingPoints.Clear();
                    drawingPointTimes.Clear();
                    drawingStrokeStartIndices.Clear();
                    lastFingertipPoint = null;
                    displayedFingertipPoint = null;
                    displayedThumbPoint = null;
                    displayedFingertipPresence = 0f;
                    fingertipMissingSince = DateTime.MinValue;
                    lastFingertipSeenAt = DateTime.MinValue;
                    lastFingertipInferenceAt = DateTime.MinValue;
                    lastIdleHandInferenceAt = DateTime.MinValue;
                    drawingStrokeStartPending = true;
                    ResetPinchGesture(false);
                    ResetStartGestures();
                    activeDrawingHand = DrawingHand.None;
                    displayedDrawingHand = DrawingHand.None;
                    waitingForOpenPalmReleaseAfterStart = false;
                    isWaving = false;
                    drawingStatusText =
                        string.IsNullOrWhiteSpace(templateImagePath)
                            ? GetIdleInstruction()
                            : "比對圖：" +
                              Path.GetFileName(templateImagePath) +
                              "｜左右手張掌開始";
                }

                return;
            }

            if (gameState == GameState.Scoring)
                return;

            if (main == null || main.Keypoints.Count <= 10 || _ratio <= 0f)
            {
                handRaisedStart = DateTime.MinValue;

                if (gameState == GameState.Idle)
                {
                    ResetStartGestures();
                    ClearDisplayedHand();
                    drawingStatusText = GetIdleInstruction();
                }
                else
                {
                    MarkFingertipMissing();
                }

                return;
            }

            switch (gameState)
            {
                case GameState.Idle:
                    if (fingertipTracker == null)
                    {
                        drawingStatusText =
                            "指尖模型未載入｜請確認模型與 TensorFlow Lite 套件";
                    }
                    else
                    {
                        UpdateIdleStartGesture(
                            frame,
                            main,
                            DateTime.Now);
                    }

                    break;

                case GameState.Drawing:
                    PointF fingertip;

                    if (TryGetFingertipPoint(
                        frame,
                        main,
                        activeDrawingHand,
                        out fingertip))
                    {
                        if (waitingForOpenPalmReleaseAfterStart)
                        {
                            if (latestOpenPalmScore <=
                                OpenPalmReleaseScore)
                            {
                                waitingForOpenPalmReleaseAfterStart =
                                    false;
                                ResetPinchGesture(false);
                                drawingStrokeStartPending = true;
                                lastFingertipPoint = null;
                                drawingStatusText =
                                    GetHandDisplayName(
                                        activeDrawingHand) +
                                    "已進入繪圖｜用食指畫圖";
                            }
                            else
                            {
                                drawingStatusText =
                                    GetHandDisplayName(
                                        activeDrawingHand) +
                                    "開始成功｜請收起其他手指";
                            }
                        }
                        else if (UpdatePinchGesture(DateTime.Now))
                        {
                            DateTime pinchDetectedAt =
                                pinchStartedAt;
                            StopDrawing(
                                pinchDetectedAt,
                                PinchTailTrimMs);
                            break;
                        }
                        else if (IsPinchDetectionActive())
                        {
                            int progress =
                                GetPinchProgressPercent(
                                    DateTime.Now);
                            drawingStatusText =
                                "捏合完成 " +
                                progress +
                                "%｜放開可取消";
                        }
                        else
                        {
                            AddDrawingPoint(fingertip);
                            drawingStatusText =
                                "繪圖中（" +
                                GetHandDisplayName(
                                    activeDrawingHand) +
                                "食指）｜再次捏合 0.6 秒完成";
                        }
                    }
                    else
                    {
                        ResetPinchGesture(true);
                        MarkFingertipMissing();
                        drawingStatusText =
                            "正在尋找" +
                            GetHandDisplayName(
                                activeDrawingHand) +
                            "食指｜整隻手與手肘請保持在畫面內";
                    }

                    break;

                case GameState.Finished:
                case GameState.Countdown:
                case GameState.Scoring:
                    break;
            }
        }

        private void UpdateYellowTipGame(Bitmap frame)
        {
            DateTime now = DateTime.Now;

            if (gameState == GameState.Finished)
            {
                if (drawingFinishedAt != DateTime.MinValue &&
                    (now - drawingFinishedAt).TotalMilliseconds >=
                        FinishedDisplayMs)
                {
                    gameState = GameState.Idle;
                    handTrail.Clear();
                    drawingPoints.Clear();
                    drawingPointTimes.Clear();
                    drawingStrokeStartIndices.Clear();
                    drawingStrokeStartPending = true;
                    lastFingertipPoint = null;
                    displayedFingertipPoint = null;
                    displayedThumbPoint = null;
                    displayedFingertipPresence = 0f;
                    yellowDrawingStartPoint = null;
                    yellowMovedAfterStart = false;
                    yellowMissingSince = DateTime.MinValue;
                    ResetYellowTracking();
                    drawingStatusText = GetIdleInstruction();
                }

                return;
            }

            if (gameState == GameState.Scoring)
                return;

            PointF tip;
            RectangleF bounds;
            bool candidateVisible;
            bool detected =
                TryGetStableYellowTip(
                    frame,
                    now,
                    out tip,
                    out bounds,
                    out candidateVisible);

            if (!detected)
            {
                ResetYellowHold();

                if (yellowMissingSince == DateTime.MinValue)
                    yellowMissingSince = now;

                if ((now - yellowMissingSince).TotalMilliseconds >=
                    YellowMissingBreakMs)
                {
                    displayedFingertipPoint = null;
                    displayedYellowTipBounds = RectangleF.Empty;

                    if (gameState == GameState.Drawing)
                    {
                        drawingStrokeStartPending = true;
                        lastFingertipPoint = null;
                    }
                }

                drawingStatusText =
                    candidateVisible
                        ? "確認黃色筆尖中｜請短暫保持穩定"
                        : gameState == GameState.Drawing
                            ? "黃色筆尖離開畫面｜軌跡已暫停"
                            : "正在尋找黃色筆尖｜找到後停留 0.6 秒開始";
                return;
            }

            yellowMissingSince = DateTime.MinValue;
            displayedFingertipPoint = tip;
            displayedThumbPoint = null;
            displayedFingertipPresence = 1f;
            displayedDrawingHand = DrawingHand.None;
            displayedYellowTipBounds = bounds;
            lastFingertipSeenAt = now;

            if (gameState == GameState.Idle)
            {
                UpdateYellowHold(tip, now);

                int progress = GetProgressPercent(
                    yellowHoldStartedAt,
                    YellowStartHoldMs,
                    now);
                drawingStatusText =
                    "黃色筆尖開始 " +
                    progress +
                    "%｜保持不動";

                if (progress >= 100)
                    StartYellowTipDrawing(tip);

                return;
            }

            if (gameState != GameState.Drawing)
                return;

            if (!yellowMovedAfterStart)
            {
                PointF origin =
                    yellowDrawingStartPoint ?? tip;
                float dx = tip.X - origin.X;
                float dy = tip.Y - origin.Y;

                if (dx * dx + dy * dy >=
                    YellowStartMoveDistance *
                    YellowStartMoveDistance)
                {
                    yellowMovedAfterStart = true;
                    drawingStrokeStartPending = true;
                    ResetYellowHold();
                    AddYellowDrawingPoint(tip);
                }
                else
                {
                    drawingStatusText =
                        "開始成功｜移動黃色筆尖開始畫圖";
                }

                return;
            }

            AddYellowDrawingPoint(tip);
            UpdateYellowHold(tip, now);
            int finishProgress = GetProgressPercent(
                yellowHoldStartedAt,
                YellowFinishHoldMs,
                now);

            if (finishProgress >= 100)
            {
                DateTime holdStartedAt =
                    yellowHoldStartedAt;
                StopDrawing(holdStartedAt, 0);
                ResetYellowHold();
                return;
            }

            drawingStatusText =
                finishProgress > 0
                    ? "停留完成 " +
                      finishProgress +
                      "%｜移動可取消"
                    : "繪圖中（黃色筆尖）｜畫完停留 0.9 秒完成";
        }

        private bool TryGetStableYellowTip(
            Bitmap frame,
            DateTime now,
            out PointF tip,
            out RectangleF bounds,
            out bool candidateVisible)
        {
            tip = PointF.Empty;
            bounds = RectangleF.Empty;
            candidateVisible = false;
            PointF rawTip;
            RectangleF rawBounds;

            if (!yellowTipTracker.TryDetect(
                frame,
                out rawTip,
                out rawBounds))
            {
                if (yellowRawMissingSince ==
                    DateTime.MinValue)
                {
                    yellowRawMissingSince = now;
                }

                if ((now - yellowRawMissingSince)
                        .TotalMilliseconds >=
                    YellowTrackingLossGraceMs)
                {
                    yellowTrackingConfirmed = false;
                    yellowCandidateStartedAt =
                        DateTime.MinValue;
                    yellowCandidateAnchor = null;
                    filteredYellowTip = null;
                    lastYellowAcceptedAt =
                        DateTime.MinValue;
                }

                return false;
            }

            candidateVisible = true;
            yellowRawMissingSince = DateTime.MinValue;

            if (!yellowTrackingConfirmed)
            {
                if (!yellowCandidateAnchor.HasValue)
                {
                    yellowCandidateAnchor = rawTip;
                    yellowCandidateStartedAt = now;
                    return false;
                }

                float candidateDx =
                    rawTip.X - yellowCandidateAnchor.Value.X;
                float candidateDy =
                    rawTip.Y - yellowCandidateAnchor.Value.Y;

                if (candidateDx * candidateDx +
                    candidateDy * candidateDy >
                    YellowReacquireRadius *
                    YellowReacquireRadius)
                {
                    yellowCandidateAnchor = rawTip;
                    yellowCandidateStartedAt = now;
                    return false;
                }

                if ((now - yellowCandidateStartedAt)
                        .TotalMilliseconds <
                    YellowReacquireConfirmMs)
                {
                    return false;
                }

                yellowTrackingConfirmed = true;
                filteredYellowTip = rawTip;
                lastYellowAcceptedAt = now;
                yellowCandidateStartedAt = DateTime.MinValue;
                yellowCandidateAnchor = null;
            }
            else if (filteredYellowTip.HasValue)
            {
                PointF previous = filteredYellowTip.Value;
                float dx = rawTip.X - previous.X;
                float dy = rawTip.Y - previous.Y;
                float distanceSquared = dx * dx + dy * dy;
                double elapsedMilliseconds =
                    lastYellowAcceptedAt == DateTime.MinValue
                        ? 33d
                        : Math.Max(
                            1d,
                            (now - lastYellowAcceptedAt)
                                .TotalMilliseconds);
                float maximumJump =
                    Math.Min(
                        YellowMaximumAdaptiveJump,
                        YellowBaseFrameJump +
                        YellowMaximumSpeedPixelsPerSecond *
                        (float)elapsedMilliseconds /
                        1000f);

                if (distanceSquared >
                    maximumJump * maximumJump)
                {
                    // Keep the existing lock. The elapsed-time allowance
                    // grows on the next frame, so a valid fast marker can
                    // catch up without requiring a stationary reacquire.
                    return false;
                }

                if (distanceSquared <=
                    YellowJitterDeadZone *
                    YellowJitterDeadZone)
                {
                    rawTip = previous;
                }
                else
                {
                    float smoothing =
                        distanceSquared >=
                            YellowFastMotionDistance *
                            YellowFastMotionDistance
                            ? YellowFastPositionSmoothing
                            : YellowSlowPositionSmoothing;
                    rawTip = new PointF(
                        previous.X + dx * smoothing,
                        previous.Y + dy * smoothing);
                }

                filteredYellowTip = rawTip;
                lastYellowAcceptedAt = now;
            }
            else
            {
                filteredYellowTip = rawTip;
                lastYellowAcceptedAt = now;
            }

            tip = filteredYellowTip.Value;
            bounds = rawBounds;
            return true;
        }

        private void AddYellowDrawingPoint(PointF point)
        {
            if (drawingStrokeStartPending ||
                drawingPoints.Count == 0)
            {
                AddDrawingPoint(point);
                return;
            }

            PointF previous =
                drawingPoints[drawingPoints.Count - 1];
            float dx = point.X - previous.X;
            float dy = point.Y - previous.Y;
            float distance =
                (float)Math.Sqrt(dx * dx + dy * dy);
            int segmentCount =
                Math.Min(
                    YellowMaximumInterpolatedPoints,
                    Math.Max(
                        1,
                        (int)Math.Ceiling(
                            distance /
                            YellowInterpolationSpacing)));

            for (int segment = 1;
                segment <= segmentCount;
                segment++)
            {
                float amount =
                    segment / (float)segmentCount;
                AddDrawingPoint(
                    new PointF(
                        previous.X + dx * amount,
                        previous.Y + dy * amount));
            }
        }

        private void StartYellowTipDrawing(PointF tip)
        {
            StartDrawing(DrawingHand.None, null);
            waitingForOpenPalmReleaseAfterStart = false;
            yellowDrawingStartPoint = tip;
            yellowMovedAfterStart = false;
            yellowMissingSince = DateTime.MinValue;
            displayedFingertipPoint = tip;
            displayedFingertipPresence = 1f;
            ResetYellowHold();
            drawingStatusText =
                "開始成功｜移動黃色筆尖開始畫圖";
        }

        private void UpdateYellowHold(
            PointF point,
            DateTime now)
        {
            if (!yellowHoldAnchor.HasValue)
            {
                yellowHoldAnchor = point;
                yellowHoldStartedAt = now;
                return;
            }

            float dx =
                point.X - yellowHoldAnchor.Value.X;
            float dy =
                point.Y - yellowHoldAnchor.Value.Y;

            if (dx * dx + dy * dy >
                YellowHoldRadius * YellowHoldRadius)
            {
                yellowHoldAnchor = point;
                yellowHoldStartedAt = now;
            }
        }

        private void ResetYellowHold()
        {
            yellowHoldAnchor = null;
            yellowHoldStartedAt = DateTime.MinValue;
        }

        private void ResetYellowTracking()
        {
            yellowTipTracker.Reset();
            yellowTrackingConfirmed = false;
            yellowCandidateStartedAt = DateTime.MinValue;
            yellowRawMissingSince = DateTime.MinValue;
            lastYellowAcceptedAt = DateTime.MinValue;
            yellowCandidateAnchor = null;
            filteredYellowTip = null;
            yellowMissingSince = DateTime.MinValue;
            displayedYellowTipBounds = RectangleF.Empty;
            ResetYellowHold();
        }

        private void UpdateIdleStartGesture(
            Bitmap frame,
            Detection person,
            DateTime now)
        {
            if (lastIdleHandInferenceAt != DateTime.MinValue &&
                (now - lastIdleHandInferenceAt).TotalMilliseconds <
                    IdleHandInferenceIntervalMs)
            {
                return;
            }

            lastIdleHandInferenceAt = now;

            FingertipResult leftResult;
            FingertipResult rightResult;
            bool leftDetected = TryDetectHand(
                frame,
                person,
                DrawingHand.Left,
                out leftResult);
            bool rightDetected = TryDetectHand(
                frame,
                person,
                DrawingHand.Right,
                out rightResult);

            bool leftConfirmed = UpdateStartGestureCandidate(
                DrawingHand.Left,
                leftDetected ? leftResult : null,
                now);
            bool rightConfirmed = UpdateStartGestureCandidate(
                DrawingHand.Right,
                rightDetected ? rightResult : null,
                now);

            if (leftConfirmed)
            {
                StartDrawing(DrawingHand.Left, leftResult);
                return;
            }

            if (rightConfirmed)
            {
                StartDrawing(DrawingHand.Right, rightResult);
                return;
            }

            FingertipResult displayResult = null;
            DrawingHand displayHand = DrawingHand.None;

            if (leftStartGestureAt != DateTime.MinValue &&
                leftDetected)
            {
                displayResult = leftResult;
                displayHand = DrawingHand.Left;
            }
            else if (rightStartGestureAt != DateTime.MinValue &&
                rightDetected)
            {
                displayResult = rightResult;
                displayHand = DrawingHand.Right;
            }
            else if (leftDetected &&
                (!rightDetected ||
                 leftResult.HandPresence >=
                    rightResult.HandPresence))
            {
                displayResult = leftResult;
                displayHand = DrawingHand.Left;
            }
            else if (rightDetected)
            {
                displayResult = rightResult;
                displayHand = DrawingHand.Right;
            }

            if (displayResult != null)
                SetDisplayedHandResult(displayHand, displayResult);
            else
                ClearDisplayedHand();

            DateTime activeStart = GetStartGestureAt(displayHand);

            if (activeStart != DateTime.MinValue)
            {
                int progress = GetProgressPercent(
                    activeStart,
                    OpenPalmStartDelayMs,
                    now);
                drawingStatusText =
                    GetHandDisplayName(displayHand) +
                    "張掌開始 " +
                    progress +
                    "%";
            }
            else
            {
                drawingStatusText = GetIdleInstruction();
            }
        }

        private bool UpdateStartGestureCandidate(
            DrawingHand hand,
            FingertipResult result,
            DateTime now)
        {
            DateTime startedAt = GetStartGestureAt(hand);

            if (result == null)
            {
                DateTime lastSeenAt =
                    GetStartGestureLastSeenAt(hand);

                if (lastSeenAt == DateTime.MinValue ||
                    (now - lastSeenAt).TotalMilliseconds >
                        StartGestureMissingGraceMs)
                {
                    SetStartGestureAt(hand, DateTime.MinValue);
                    SetStartGestureLastSeenAt(
                        hand,
                        DateTime.MinValue);
                }

                return false;
            }

            SetStartGestureLastSeenAt(hand, now);

            if (result.OpenPalmScore >= OpenPalmStartScore)
            {
                if (startedAt == DateTime.MinValue)
                {
                    startedAt = now;
                    SetStartGestureAt(hand, startedAt);
                }

                return
                    (now - startedAt).TotalMilliseconds >=
                    OpenPalmStartDelayMs;
            }

            if (result.OpenPalmScore <= OpenPalmReleaseScore)
                SetStartGestureAt(hand, DateTime.MinValue);

            return false;
        }

        private void ResetStartGestures()
        {
            leftStartGestureAt = DateTime.MinValue;
            rightStartGestureAt = DateTime.MinValue;
            leftStartGestureLastSeenAt = DateTime.MinValue;
            rightStartGestureLastSeenAt = DateTime.MinValue;
        }

        private DateTime GetStartGestureAt(DrawingHand hand)
        {
            if (hand == DrawingHand.Left)
                return leftStartGestureAt;

            if (hand == DrawingHand.Right)
                return rightStartGestureAt;

            return DateTime.MinValue;
        }

        private void SetStartGestureAt(
            DrawingHand hand,
            DateTime value)
        {
            if (hand == DrawingHand.Left)
                leftStartGestureAt = value;
            else if (hand == DrawingHand.Right)
                rightStartGestureAt = value;
        }

        private DateTime GetStartGestureLastSeenAt(
            DrawingHand hand)
        {
            if (hand == DrawingHand.Left)
                return leftStartGestureLastSeenAt;

            if (hand == DrawingHand.Right)
                return rightStartGestureLastSeenAt;

            return DateTime.MinValue;
        }

        private void SetStartGestureLastSeenAt(
            DrawingHand hand,
            DateTime value)
        {
            if (hand == DrawingHand.Left)
                leftStartGestureLastSeenAt = value;
            else if (hand == DrawingHand.Right)
                rightStartGestureLastSeenAt = value;
        }

        private int GetProgressPercent(
            DateTime startedAt,
            int delayMs,
            DateTime now)
        {
            if (startedAt == DateTime.MinValue || delayMs <= 0)
                return 0;

            double progress =
                (now - startedAt).TotalMilliseconds /
                delayMs;

            return (int)Math.Max(
                0d,
                Math.Min(100d, progress * 100d));
        }

        private string GetIdleInstruction()
        {
            return "黃色筆尖停留 0.6 秒開始｜畫完停留 0.9 秒完成";
        }

        private string GetHandDisplayName(DrawingHand hand)
        {
            if (hand == DrawingHand.Left)
                return "左手";

            if (hand == DrawingHand.Right)
                return "右手";

            return "手";
        }

        private void SetDisplayedHandResult(
            DrawingHand hand,
            FingertipResult result)
        {
            displayedDrawingHand = hand;
            displayedFingertipPoint = result.IndexTip;
            displayedThumbPoint = result.ThumbTip;
            displayedFingertipPresence = result.HandPresence;
            latestPinchRatio = result.PinchRatio;
            latestOpenPalmScore = result.OpenPalmScore;
            lastFingertipSeenAt = DateTime.Now;
        }

        private void ClearDisplayedHand()
        {
            displayedDrawingHand = DrawingHand.None;
            displayedFingertipPoint = null;
            displayedThumbPoint = null;
            displayedFingertipPresence = 0f;
            latestPinchRatio = float.MaxValue;
            latestOpenPalmScore = 0f;
        }

        private bool TryDetectHand(
            Bitmap frame,
            Detection person,
            DrawingHand hand,
            out FingertipResult result)
        {
            result = null;

            if (fingertipTracker == null ||
                frame == null ||
                person == null ||
                person.Keypoints.Count <= 10 ||
                hand == DrawingHand.None)
            {
                latestHandInferenceResult = "NO ROI";
                return false;
            }

            int elbowIndex =
                hand == DrawingHand.Left ? 7 : 8;
            int wristIndex =
                hand == DrawingHand.Left ? 9 : 10;
            Keypoint elbowKeypoint =
                person.Keypoints[elbowIndex];
            Keypoint wristKeypoint =
                person.Keypoints[wristIndex];
            bool elbowReliable =
                elbowKeypoint.Score >=
                    MinimumPoseHandKeypointScore;
            bool wristReliable =
                wristKeypoint.Score >=
                    MinimumPoseHandKeypointScore;
            DateTime now = DateTime.Now;
            HandTrackingAnchor anchor =
                GetHandTrackingAnchor(hand);
            bool canUseTrackingFallback =
                gameState == GameState.Drawing &&
                hand == activeDrawingHand &&
                anchor.IsRecent(now);
            PointF elbow;
            PointF wrist;

            if (elbowReliable && wristReliable)
            {
                elbow =
                    KeypointToImagePoint(elbowKeypoint);
                wrist =
                    KeypointToImagePoint(wristKeypoint);
            }
            else if (canUseTrackingFallback &&
                wristReliable)
            {
                wrist =
                    KeypointToImagePoint(wristKeypoint);
                elbow = new PointF(
                    wrist.X +
                    anchor.Elbow.X -
                    anchor.Wrist.X,
                    wrist.Y +
                    anchor.Elbow.Y -
                    anchor.Wrist.Y);
            }
            else if (canUseTrackingFallback)
            {
                wrist = anchor.Wrist;
                elbow = anchor.Elbow;
            }
            else
            {
                latestHandInferenceResult = "NO ROI";
                return false;
            }

            PointF forearmOffset = new PointF(
                elbow.X - wrist.X,
                elbow.Y - wrist.Y);

            bool consumed =
                TryConsumeHandInference(
                    hand,
                    anchor,
                    out result);

            QueueHandInference(
                frame,
                hand,
                wrist,
                elbow,
                forearmOffset);

            return consumed;
        }

        private bool TryConsumeHandInference(
            DrawingHand hand,
            HandTrackingAnchor anchor,
            out FingertipResult result)
        {
            result = null;
            HandInferenceSnapshot snapshot =
                latestHandSnapshot;

            if (snapshot == null ||
                snapshot.Hand != hand)
            {
                return false;
            }

            DateTime lastConsumed =
                hand == DrawingHand.Left
                    ? lastConsumedLeftHandAt
                    : lastConsumedRightHandAt;

            if (snapshot.CompletedAt <= lastConsumed)
            {
                if (snapshot.Detected &&
                    snapshot.Result != null &&
                    (DateTime.Now -
                        snapshot.CompletedAt)
                        .TotalMilliseconds <
                    FingertipMarkerVisibleMs)
                {
                    result = snapshot.Result;
                    return true;
                }

                return false;
            }

            if (hand == DrawingHand.Left)
                lastConsumedLeftHandAt =
                    snapshot.CompletedAt;
            else
                lastConsumedRightHandAt =
                    snapshot.CompletedAt;

            latestHandInferenceMilliseconds =
                snapshot.Milliseconds;

            if (snapshot.Error != null)
            {
                latestHandInferenceResult = "ERROR";
                Debug.WriteLine(
                    "Hand inference error (" +
                    GetHandDisplayName(hand) +
                    "): " +
                    snapshot.Error);
                return false;
            }

            latestHandInferenceResult =
                snapshot.Detected
                    ? "FOUND"
                    : "LOST";

            if (!snapshot.Detected ||
                snapshot.Result == null)
            {
                return false;
            }

            result = snapshot.Result;
            anchor.Wrist = result.Wrist;
            anchor.Elbow = new PointF(
                result.Wrist.X +
                snapshot.ForearmOffset.X,
                result.Wrist.Y +
                snapshot.ForearmOffset.Y);
            anchor.UpdatedAt =
                snapshot.CompletedAt;
            return true;
        }

        private void QueueHandInference(
            Bitmap frame,
            DrawingHand hand,
            PointF wrist,
            PointF elbow,
            PointF forearmOffset)
        {
            if (!grabImage ||
                fingertipTracker == null ||
                !ShouldScheduleHand(hand) ||
                Interlocked.CompareExchange(
                    ref handInferenceBusy,
                    1,
                    0) != 0)
            {
                return;
            }

            Bitmap inferenceFrame = null;

            try
            {
                inferenceFrame = frame.Clone(
                    new Rectangle(
                        0,
                        0,
                        frame.Width,
                        frame.Height),
                    PixelFormat.Format24bppRgb);
                float minimumPresence =
                    gameState == GameState.Drawing &&
                    hand == activeDrawingHand
                        ? MinimumDrawingHandPresence
                        : MinimumIdleHandPresence;

                if (gameState == GameState.Idle)
                {
                    nextIdleHandRequest =
                        hand == DrawingHand.Left
                            ? DrawingHand.Right
                            : DrawingHand.Left;
                }

                Bitmap ownedFrame = inferenceFrame;
                inferenceFrame = null;
                handInferenceTask = Task.Run(() =>
                {
                    Stopwatch stopwatch =
                        Stopwatch.StartNew();
                    FingertipResult detectedResult = null;
                    bool detected = false;
                    Exception error = null;

                    try
                    {
                        detected =
                            fingertipTracker.TryDetect(
                                ownedFrame,
                                wrist,
                                elbow,
                                minimumPresence,
                                out detectedResult);
                    }
                    catch (Exception ex)
                    {
                        error = ex;
                    }
                    finally
                    {
                        stopwatch.Stop();
                        ownedFrame.Dispose();
                    }

                    latestHandSnapshot =
                        new HandInferenceSnapshot(
                            hand,
                            detectedResult,
                            detected,
                            DateTime.Now,
                            forearmOffset,
                            stopwatch.Elapsed
                                .TotalMilliseconds,
                            error);
                    latestHandInferenceMilliseconds =
                        stopwatch.Elapsed
                            .TotalMilliseconds;
                    latestHandInferenceResult =
                        error != null
                            ? "ERROR"
                            : detected
                                ? "FOUND"
                                : "LOST";
                    Interlocked.Exchange(
                        ref handInferenceBusy,
                        0);
                });
            }
            catch (Exception ex)
            {
                inferenceFrame?.Dispose();
                latestHandInferenceResult = "ERROR";
                Debug.WriteLine(
                    "Unable to queue hand inference: " +
                    ex);
                Interlocked.Exchange(
                    ref handInferenceBusy,
                    0);
            }
        }

        private bool ShouldScheduleHand(
            DrawingHand hand)
        {
            if (gameState == GameState.Drawing)
                return hand == activeDrawingHand;

            if (gameState == GameState.Idle)
                return hand == nextIdleHandRequest;

            return false;
        }

        private void WaitForHandInference(
            int timeoutMilliseconds)
        {
            Task task = handInferenceTask;

            if (task == null)
                return;

            try
            {
                task.Wait(timeoutMilliseconds);
            }
            catch (AggregateException ex)
            {
                Debug.WriteLine(
                    "Hand worker shutdown error: " +
                    ex.Flatten());
            }
        }

        private HandTrackingAnchor GetHandTrackingAnchor(
            DrawingHand hand)
        {
            return hand == DrawingHand.Left
                ? leftHandTracking
                : rightHandTracking;
        }

        private bool TryGetFingertipPoint(
            Bitmap frame,
            Detection person,
            DrawingHand hand,
            out PointF fingertip)
        {
            fingertip = PointF.Empty;

            if (fingertipTracker == null ||
                frame == null ||
                person == null ||
                person.Keypoints.Count <= 10)
            {
                return false;
            }

            DateTime now = DateTime.Now;

            if (lastFingertipInferenceAt != DateTime.MinValue &&
                (now - lastFingertipInferenceAt).TotalMilliseconds <
                    FingertipInferenceIntervalMs)
            {
                if (lastFingertipPoint.HasValue &&
                    lastFingertipSeenAt != DateTime.MinValue &&
                    (now - lastFingertipSeenAt).TotalMilliseconds <
                        FingertipMarkerVisibleMs)
                {
                    fingertip = lastFingertipPoint.Value;
                    return true;
                }

                return false;
            }

            lastFingertipInferenceAt = now;

            FingertipResult result;

            if (!TryDetectHand(
                frame,
                person,
                hand,
                out result))
            {
                return false;
            }

            PointF detected = result.IndexTip;
            PointF? previousThumb = displayedThumbPoint;

            if (!waitingForOpenPalmReleaseAfterStart &&
                lastFingertipPoint.HasValue &&
                previousThumb.HasValue &&
                result.PinchRatio <= PinchStartRatio &&
                (IsPinchDetectionActive() ||
                 latestPinchRatio >= PinchReleaseRatio))
            {
                PointF previousIndex =
                    lastFingertipPoint.Value;
                float indexDx = detected.X - previousIndex.X;
                float indexDy = detected.Y - previousIndex.Y;
                float thumbDx =
                    result.ThumbTip.X - previousThumb.Value.X;
                float thumbDy =
                    result.ThumbTip.Y - previousThumb.Value.Y;
                float indexMovementSquared =
                    indexDx * indexDx + indexDy * indexDy;
                float thumbMovementSquared =
                    thumbDx * thumbDx + thumbDy * thumbDy;
                float suspiciousMovement = Math.Max(
                    18f,
                    GetMaximumFingertipJump(person) * 0.25f);

                if (indexMovementSquared >=
                        suspiciousMovement * suspiciousMovement &&
                    thumbMovementSquared * 2.25f <
                        indexMovementSquared)
                {
                    SetDisplayedHandResult(hand, result);
                    displayedFingertipPoint = previousIndex;
                    lastFingertipSeenAt = now;
                    fingertipMissingSince = DateTime.MinValue;
                    fingertip = previousIndex;
                    return true;
                }
            }

            if (lastFingertipPoint.HasValue &&
                lastFingertipSeenAt != DateTime.MinValue &&
                (now - lastFingertipSeenAt).TotalMilliseconds <
                    FingertipMissingBreakMs)
            {
                PointF previous = lastFingertipPoint.Value;
                float dx = detected.X - previous.X;
                float dy = detected.Y - previous.Y;
                float maximumJump =
                    GetMaximumFingertipJump(person);

                if (dx * dx + dy * dy >
                    maximumJump * maximumJump)
                {
                    return false;
                }

                detected = new PointF(
                    previous.X +
                    (detected.X - previous.X) * FingertipSmoothingFactor,
                    previous.Y +
                    (detected.Y - previous.Y) * FingertipSmoothingFactor);
            }

            lastFingertipPoint = detected;
            SetDisplayedHandResult(hand, result);
            displayedFingertipPoint = detected;
            lastFingertipSeenAt = now;
            fingertipMissingSince = DateTime.MinValue;
            fingertip = detected;

            return true;
        }

        private bool UpdatePinchGesture(DateTime now)
        {
            if (!pinchInProgress)
            {
                if (pinchCandidateStartedAt == DateTime.MinValue)
                {
                    if (latestPinchRatio > PinchStartRatio)
                        return false;

                    pinchCandidateStartedAt = now;
                    drawingStrokeStartPending = true;
                    return false;
                }

                if (latestPinchRatio >= PinchReleaseRatio)
                {
                    ResetPinchGesture(true);
                    return false;
                }

                if ((now - pinchCandidateStartedAt)
                        .TotalMilliseconds < PinchPreconfirmMs)
                {
                    return false;
                }

                pinchInProgress = true;
                pinchStartedAt = pinchCandidateStartedAt;
                pinchCandidateStartedAt = DateTime.MinValue;
                drawingStrokeStartPending = true;
            }
            else if (latestPinchRatio >= PinchReleaseRatio)
            {
                ResetPinchGesture(true);
                return false;
            }

            return
                (now - pinchStartedAt).TotalMilliseconds >=
                PinchFinishDelayMs;
        }

        private int GetPinchProgressPercent(DateTime now)
        {
            DateTime progressStartedAt =
                pinchInProgress
                    ? pinchStartedAt
                    : pinchCandidateStartedAt;

            if (progressStartedAt == DateTime.MinValue)
                return 0;

            double progress =
                (now - progressStartedAt).TotalMilliseconds /
                PinchFinishDelayMs;

            return (int)Math.Max(
                0d,
                Math.Min(100d, progress * 100d));
        }

        private void ResetPinchGesture(bool startNewStroke)
        {
            bool wasPinching = IsPinchDetectionActive();

            pinchInProgress = false;
            pinchCandidateStartedAt = DateTime.MinValue;
            pinchStartedAt = DateTime.MinValue;
            latestPinchRatio = float.MaxValue;

            if (startNewStroke && wasPinching)
            {
                drawingStrokeStartPending = true;
                lastFingertipPoint = null;
            }
        }

        private bool IsPinchDetectionActive()
        {
            return pinchInProgress ||
                pinchCandidateStartedAt != DateTime.MinValue;
        }

        private PointF KeypointToImagePoint(Keypoint keypoint)
        {
            return new PointF(
                (keypoint.X - _padX) / _ratio,
                (keypoint.Y - _padY) / _ratio);
        }

        private float GetMaximumFingertipJump(Detection person)
        {
            if (person.Keypoints.Count <= 6 ||
                person.Keypoints[5].Score < 0.5f ||
                person.Keypoints[6].Score < 0.5f)
                return 80f;

            PointF leftShoulder =
                KeypointToImagePoint(person.Keypoints[5]);
            PointF rightShoulder =
                KeypointToImagePoint(person.Keypoints[6]);
            float dx = rightShoulder.X - leftShoulder.X;
            float dy = rightShoulder.Y - leftShoulder.Y;
            float shoulderWidth = (float)Math.Sqrt(dx * dx + dy * dy);

            return Math.Max(80f, shoulderWidth * 0.65f);
        }

        private void MarkFingertipMissing()
        {
            if (gameState != GameState.Drawing)
                return;

            ResetPinchGesture(true);
            DateTime now = DateTime.Now;

            if (fingertipMissingSince == DateTime.MinValue)
                fingertipMissingSince = now;

            if ((now - fingertipMissingSince).TotalMilliseconds >=
                FingertipMissingBreakMs)
            {
                drawingStrokeStartPending = true;
                lastFingertipPoint = null;
            }

            if (lastFingertipSeenAt == DateTime.MinValue ||
                (now - lastFingertipSeenAt).TotalMilliseconds >=
                    FingertipMarkerVisibleMs)
            {
                ClearDisplayedHand();
            }
        }

        private void pictureBoxInvoke(Bitmap bmp)
        {
            Image old = pBox.Image;
            pBox.Image = bmp;
            // Let Windows coalesce paint requests instead of blocking the UI
            // thread until PictureBox finishes repainting each frame.
            pBox.Invalidate();
            old?.Dispose();
        }

        private void QueueDisplayImage(Bitmap bitmap)
        {
            if (bitmap == null)
                return;

            if (IsDisposed || Disposing || pBox.IsDisposed)
            {
                bitmap.Dispose();
                return;
            }

            Bitmap replaced = Interlocked.Exchange(
                ref pendingDisplayImage,
                bitmap);
            replaced?.Dispose();

            ScheduleDisplayUpdate();
        }

        private void ScheduleDisplayUpdate()
        {
            if (Interlocked.CompareExchange(
                ref displayUpdateScheduled,
                1,
                0) != 0)
            {
                return;
            }

            try
            {
                pBox.BeginInvoke(
                    new MethodInvoker(PresentLatestDisplayImage));
            }
            catch (ObjectDisposedException)
            {
                Interlocked.Exchange(ref displayUpdateScheduled, 0);
                DisposePendingDisplayImage();
            }
            catch (InvalidOperationException)
            {
                Interlocked.Exchange(ref displayUpdateScheduled, 0);
                DisposePendingDisplayImage();
            }
        }

        private void PresentLatestDisplayImage()
        {
            Bitmap latest = Interlocked.Exchange(
                ref pendingDisplayImage,
                null);

            if (latest != null)
            {
                if (IsDisposed || Disposing || pBox.IsDisposed)
                    latest.Dispose();
                else
                {
                    pictureBoxInvoke(latest);
                    RecordDisplayFrame();
                    UpdateAirDrawStatusBar();
                }
            }

            Interlocked.Exchange(ref displayUpdateScheduled, 0);

            if (Interlocked.CompareExchange(
                ref pendingDisplayImage,
                null,
                null) != null)
            {
                ScheduleDisplayUpdate();
            }
        }

        private void DisposePendingDisplayImage()
        {
            Bitmap pending = Interlocked.Exchange(
                ref pendingDisplayImage,
                null);
            pending?.Dispose();
        }

        private void CreateTensorFromFLIR(IManagedImage img, ref DenseTensor<float> tensor)
        {
            Span<float> span = tensor.Buffer.Span;
            span.Clear();

            int hw = yoloImgWidth * yoloImgHeight;
            int rOffset = 0;
            int gOffset = hw;
            int bOffset = hw * 2;

            const float INV255 = 1.0f / 255.0f;

            unsafe
            {
                byte* src = (byte*)img.DataPtr;
                int stride = (int)img.Stride;

                for (int y = 0; y < newH; y++)
                {
                    int srcY = (int)(y / _ratio);
                    byte* srcRow = src + srcY * stride;

                    int tensorY = y + _padY;
                    int tensorRow = tensorY * yoloImgWidth;

                    for (int x = 0; x < newW; x++)
                    {
                        int srcX = (int)(x / _ratio);
                        int srcIdx = srcX * 3;

                        byte b = srcRow[srcIdx];
                        byte g = srcRow[srcIdx + 1];
                        byte r = srcRow[srcIdx + 2];

                        int tensorX = x + _padX;
                        int idx = tensorRow + tensorX;

                        span[rOffset + idx] = r * INV255;
                        span[gOffset + idx] = g * INV255;
                        span[bOffset + idx] = b * INV255;
                    }
                }
            }
        }

        private void CreateTensorFromBitmap(
            Bitmap bitmap,
            ref DenseTensor<float> tensor)
        {
            Span<float> span = tensor.Buffer.Span;
            span.Clear();

            int hw = yoloImgWidth * yoloImgHeight;
            int rOffset = 0;
            int gOffset = hw;
            int bOffset = hw * 2;
            const float inverse255 = 1.0f / 255.0f;

            Rectangle bounds =
                new Rectangle(
                    0,
                    0,
                    bitmap.Width,
                    bitmap.Height);
            BitmapData bitmapData =
                bitmap.LockBits(
                    bounds,
                    ImageLockMode.ReadOnly,
                    PixelFormat.Format24bppRgb);

            try
            {
                unsafe
                {
                    byte* sourceBase =
                        (byte*)bitmapData.Scan0
                            .ToPointer();
                    int sourceStride =
                        bitmapData.Stride;

                    for (int y = 0; y < newH; y++)
                    {
                        int sourceY =
                            Math.Min(
                                bitmap.Height - 1,
                                (int)(y / _ratio));
                        byte* sourceRow =
                            sourceStride >= 0
                                ? sourceBase +
                                  sourceY *
                                  sourceStride
                                : sourceBase +
                                  (bitmap.Height -
                                   1 -
                                   sourceY) *
                                  -sourceStride;
                        int tensorRow =
                            (y + _padY) *
                            yoloImgWidth;

                        for (int x = 0;
                            x < newW;
                            x++)
                        {
                            int sourceX =
                                Math.Min(
                                    bitmap.Width - 1,
                                    (int)(x / _ratio));
                            int sourceIndex =
                                sourceX * 3;
                            int tensorIndex =
                                tensorRow +
                                x +
                                _padX;

                            span[rOffset +
                                tensorIndex] =
                                sourceRow[
                                    sourceIndex +
                                    2] *
                                inverse255;
                            span[gOffset +
                                tensorIndex] =
                                sourceRow[
                                    sourceIndex +
                                    1] *
                                inverse255;
                            span[bOffset +
                                tensorIndex] =
                                sourceRow[
                                    sourceIndex] *
                                inverse255;
                        }
                    }
                }
            }
            finally
            {
                bitmap.UnlockBits(bitmapData);
            }
        }

        private void RenderDisplayImage(
            Bitmap copy,
            List<Detection> boxes,
            Detection main,
            double poseInferenceMilliseconds)
        {
            UpdateGame(main, copy);

            using (Graphics g = Graphics.FromImage(copy))
            using (Pen mainPen = new Pen(Color.Lime, 3))
            using (Pen otherPen = new Pen(
                Color.FromArgb(120, 200, 200, 200),
                1))
            using (Pen boxPen = new Pen(Color.Red, 2))
            using (Pen eyePen = new Pen(Color.Magenta, 3))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.CompositingQuality = CompositingQuality.HighSpeed;
                g.PixelOffsetMode = PixelOffsetMode.HighSpeed;
                g.InterpolationMode = InterpolationMode.Low;

                mainPen.StartCap = LineCap.Round;
                mainPen.EndCap = LineCap.Round;
                mainPen.LineJoin = LineJoin.Round;
                otherPen.StartCap = LineCap.Round;
                otherPen.EndCap = LineCap.Round;
                otherPen.LineJoin = LineJoin.Round;

                // 自訂骨架
                int[,] skeleton = new int[,]
                {
                    {1,0}, {2,0},
                    {5,6},
                    {5,7},{7,9},
                    {6,8},{8,10},
                    {5,11},{6,12},
                    {11,12},
                    {11,13},{13,15},
                    {12,14},{14,16}
                };

                // 軌跡（畫在最上層前）
                DrawHandTrail(g);

                foreach (var b in boxes)
                {
                    bool isMain = (main != null && b == main);
                    var pen = isMain ? mainPen : otherPen;

                    float x = (b.X - _padX) / _ratio;
                    float y = (b.Y - _padY) / _ratio;
                    float w = b.W / _ratio;
                    float h = b.H / _ratio;

                    float left = x - w / 2;
                    float top = y - h / 2;

                    if (isMain) g.DrawRectangle(boxPen, left, top, w, h);

                    // 🔥 彩色發光骨架（穩定版）
                    Color[] jointColors = new Color[]
                    {
                        Color.White, Color.White, Color.White, Color.White, Color.White,
                        Color.Blue, Color.Blue,
                        Color.Cyan, Color.Cyan,
                        Color.Green, Color.Green,
                        Color.Lime, Color.Lime,
                        Color.Yellow, Color.Yellow,
                        Color.Orange, Color.Orange
                    };

                    for (int i = 0; i < skeleton.GetLength(0); i++)
                    {
                        var p1 = b.Keypoints[skeleton[i, 0]];
                        var p2 = b.Keypoints[skeleton[i, 1]];

                        if (p1.Score > 0.5f && p2.Score > 0.5f)
                        {
                            float x1 = (p1.X - _padX) / _ratio;
                            float y1 = (p1.Y - _padY) / _ratio;
                            float x2 = (p2.X - _padX) / _ratio;
                            float y2 = (p2.Y - _padY) / _ratio;

                            var baseColor = isMain ? jointColors[skeleton[i, 1]] : Color.Gray;

                            using (Pen glow = new Pen(Color.FromArgb(120, baseColor), isMain ? 10 : 4))
                            {
                                glow.StartCap = LineCap.Round;
                                glow.EndCap = LineCap.Round;
                                g.DrawLine(glow, x1, y1, x2, y2);
                            }

                            using (Pen core = new Pen(baseColor, isMain ? 3 : 1))
                            {
                                core.StartCap = LineCap.Round;
                                core.EndCap = LineCap.Round;
                                g.DrawLine(core, x1, y1, x2, y2);
                            }
                        }
                    }

                    foreach (var kp in b.Keypoints)
                    {
                        if (kp.Score > 0.5f)
                        {
                            float px = (kp.X - _padX) / _ratio;
                            float py = (kp.Y - _padY) / _ratio;

                            int size = isMain ? 6 : 3;

                            using (Brush glow = new SolidBrush(Color.FromArgb(120, 255, 255, 255)))
                            {
                                g.FillEllipse(glow, px - size, py - size, size * 2, size * 2);
                            }

                            Brush core = isMain ? Brushes.White : Brushes.Gray;
                            g.FillEllipse(core, px - size / 2, py - size / 2, size, size);
                        }
                    }

                    var eyeL = b.Keypoints[1];
                    var eyeR = b.Keypoints[2];
                    var nose = b.Keypoints[0];

                    if (eyeL.Score > 0.5f && nose.Score > 0.5f)
                    {
                        g.DrawLine(eyePen,
                            (eyeL.X - _padX) / _ratio,
                            (eyeL.Y - _padY) / _ratio,
                            (nose.X - _padX) / _ratio,
                            (nose.Y - _padY) / _ratio);
                    }

                    if (eyeR.Score > 0.5f && nose.Score > 0.5f)
                    {
                        g.DrawLine(eyePen,
                            (eyeR.X - _padX) / _ratio,
                            (eyeR.Y - _padY) / _ratio,
                            (nose.X - _padX) / _ratio,
                            (nose.Y - _padY) / _ratio);
                    }

                    var lShoulder = b.Keypoints[5];
                    var rShoulder = b.Keypoints[6];

                    if (nose.Score > 0.5f && lShoulder.Score > 0.5f && rShoulder.Score > 0.5f)
                    {
                        float midX = (lShoulder.X + rShoulder.X) / 2;
                        float midY = (lShoulder.Y + rShoulder.Y) / 2;

                        float x1 = (nose.X - _padX) / _ratio;
                        float y1 = (nose.Y - _padY) / _ratio;
                        float x2 = (midX - _padX) / _ratio;
                        float y2 = (midY - _padY) / _ratio;

                        using (Pen torsoPen = new Pen(
                            Color.Cyan,
                            isMain ? 3 : 1))
                        {
                            g.DrawLine(
                                torsoPen,
                                x1,
                                y1,
                                x2,
                                y2);
                        }
                    }
                }

                // HELLO UI
                if (isWaving)
                {
                    using (Brush helloBackground =
                        new SolidBrush(Color.FromArgb(120, 0, 0, 0)))
                    using (Font helloFont = new Font(
                        "Arial",
                        32,
                        FontStyle.Bold))
                    {
                        g.FillRectangle(
                            helloBackground,
                            5,
                            5,
                            260,
                            60);
                        g.DrawString(
                            " HELLO!",
                            helloFont,
                            Brushes.Yellow,
                            new PointF(10, 100));
                    }
                }

                DrawFingertipMarker(g);
                DrawMagicAnimation(g, copy.Width, copy.Height);
            }
        }

        private unsafe Bitmap CopyManagedImageToBitmap(
            IManagedImage image)
        {
            int width = (int)image.Width;
            int height = (int)image.Height;
            int sourceStride = (int)image.Stride;
            int rowBytes = width * 3;

            Bitmap bitmap = new Bitmap(
                width,
                height,
                PixelFormat.Format24bppRgb);

            try
            {
                Rectangle bounds = new Rectangle(
                    0,
                    0,
                    width,
                    height);
                BitmapData bitmapData = bitmap.LockBits(
                    bounds,
                    ImageLockMode.WriteOnly,
                    PixelFormat.Format24bppRgb);

                try
                {
                    byte* sourceBase =
                        (byte*)image.DataPtr.ToPointer();
                    byte* destinationBase =
                        (byte*)bitmapData.Scan0.ToPointer();
                    int destinationStride = bitmapData.Stride;

                    for (int y = 0; y < height; y++)
                    {
                        byte* sourceRow =
                            sourceStride >= 0
                                ? sourceBase + y * sourceStride
                                : sourceBase +
                                  (height - 1 - y) *
                                  -sourceStride;
                        byte* destinationRow =
                            destinationStride >= 0
                                ? destinationBase +
                                  y * destinationStride
                                : destinationBase +
                                  (height - 1 - y) *
                                  -destinationStride;

                        Buffer.MemoryCopy(
                            sourceRow,
                            destinationRow,
                            Math.Abs(destinationStride),
                            rowBytes);
                    }
                }
                finally
                {
                    bitmap.UnlockBits(bitmapData);
                }

                return bitmap;
            }
            catch
            {
                bitmap.Dispose();
                throw;
            }
        }

        private void DrawHandTrail(Graphics graphics)
        {
            if (handTrail.Count == 0)
                return;

            using (Pen trailPen = new Pen(Color.Yellow, 4f))
            using (Brush pointBrush = new SolidBrush(Color.Yellow))
            {
                trailPen.StartCap = LineCap.Round;
                trailPen.EndCap = LineCap.Round;
                trailPen.LineJoin = LineJoin.Round;

                for (int strokeIndex = 0;
                    strokeIndex < drawingStrokeStartIndices.Count;
                    strokeIndex++)
                {
                    int start = drawingStrokeStartIndices[strokeIndex];
                    int end = Math.Min(
                        strokeIndex + 1 < drawingStrokeStartIndices.Count
                            ? drawingStrokeStartIndices[strokeIndex + 1]
                            : handTrail.Count,
                        handTrail.Count);
                    int count = end - start;

                    if (start < 0 ||
                        start >= handTrail.Count ||
                        count <= 0)
                    {
                        continue;
                    }

                    if (count == 1)
                    {
                        PointF point = handTrail[start];
                        graphics.FillEllipse(
                            pointBrush,
                            point.X - 2f,
                            point.Y - 2f,
                            4f,
                            4f);
                    }
                    else
                    {
                        PointF[] stroke = new PointF[count];
                        handTrail.CopyTo(start, stroke, 0, count);
                        graphics.DrawLines(trailPen, stroke);
                    }
                }
            }
        }

        private void DrawFingertipMarker(Graphics graphics)
        {
            if ((gameState != GameState.Idle &&
                 gameState != GameState.Drawing) ||
                !displayedFingertipPoint.HasValue)
            {
                return;
            }

            PointF point = displayedFingertipPoint.Value;

            if (yellowTipMode)
            {
                using (Pen markerPen = new Pen(Color.Gold, 4f))
                using (Pen boundsPen = new Pen(
                    Color.FromArgb(180, 255, 215, 0),
                    2f))
                using (Brush centerBrush =
                    new SolidBrush(Color.White))
                using (Font markerFont = new Font(
                    "Microsoft JhengHei UI",
                    11f,
                    FontStyle.Bold))
                {
                    graphics.DrawRectangle(
                        boundsPen,
                        displayedYellowTipBounds.X,
                        displayedYellowTipBounds.Y,
                        displayedYellowTipBounds.Width,
                        displayedYellowTipBounds.Height);
                    graphics.DrawEllipse(
                        markerPen,
                        point.X - 12f,
                        point.Y - 12f,
                        24f,
                        24f);
                    graphics.FillEllipse(
                        centerBrush,
                        point.X - 3f,
                        point.Y - 3f,
                        6f,
                        6f);
                    graphics.DrawString(
                        "黃色筆尖",
                        markerFont,
                        Brushes.Gold,
                        point.X + 15f,
                        point.Y - 14f);

                    int delay =
                        gameState == GameState.Idle
                            ? YellowStartHoldMs
                            : YellowFinishHoldMs;
                    int progress = GetProgressPercent(
                        yellowHoldStartedAt,
                        delay,
                        DateTime.Now);

                    if (progress > 0)
                    {
                        using (Pen progressPen =
                            new Pen(Color.Lime, 4f))
                        {
                            graphics.DrawArc(
                                progressPen,
                                point.X - 17f,
                                point.Y - 17f,
                                34f,
                                34f,
                                -90f,
                                360f * progress / 100f);
                        }
                    }
                }

                return;
            }

            float gestureProgress;
            bool pinchGesture;
            bool showGestureProgress =
                TryGetDisplayedGestureProgress(
                    out gestureProgress,
                    out pinchGesture);

            using (Pen outline = new Pen(Color.Cyan, 3f))
            using (Pen thumbOutline = new Pen(Color.Magenta, 3f))
            using (Pen pinchLine = new Pen(Color.Orange, 2f))
            using (Pen progressPen = new Pen(
                pinchGesture ? Color.Orange : Color.Lime,
                4f))
            using (Brush center = new SolidBrush(Color.White))
            using (Font font = new Font(
                "Microsoft JhengHei UI",
                10f,
                FontStyle.Bold))
            {
                graphics.DrawEllipse(
                    outline,
                    point.X - 9f,
                    point.Y - 9f,
                    18f,
                    18f);
                graphics.FillEllipse(
                    center,
                    point.X - 3f,
                    point.Y - 3f,
                    6f,
                    6f);
                graphics.DrawString(
                    GetHandDisplayName(displayedDrawingHand) +
                    "食指 " +
                    (displayedFingertipPresence * 100f).ToString("0") +
                    "%",
                    font,
                    Brushes.Cyan,
                    point.X + 12f,
                    point.Y - 12f);

                if (displayedThumbPoint.HasValue)
                {
                    PointF thumb = displayedThumbPoint.Value;

                    graphics.DrawEllipse(
                        thumbOutline,
                        thumb.X - 7f,
                        thumb.Y - 7f,
                        14f,
                        14f);

                    if (showGestureProgress && pinchGesture)
                    {
                        graphics.DrawLine(
                            pinchLine,
                            thumb,
                            point);
                    }
                }

                if (showGestureProgress)
                {
                    graphics.DrawArc(
                        progressPen,
                        point.X - 14f,
                        point.Y - 14f,
                        28f,
                        28f,
                        -90f,
                        360f * gestureProgress);
                }

                if (latestPinchRatio < float.MaxValue)
                {
                    graphics.DrawString(
                        "捏合 " +
                        latestPinchRatio.ToString("0.00"),
                        font,
                        showGestureProgress && pinchGesture
                            ? Brushes.Orange
                            : Brushes.Magenta,
                        point.X + 12f,
                        point.Y + 6f);
                }
            }
        }

        private bool TryGetDisplayedGestureProgress(
            out float progress,
            out bool pinchGesture)
        {
            DateTime now = DateTime.Now;

            if (gameState == GameState.Idle)
            {
                DateTime start =
                    GetStartGestureAt(displayedDrawingHand);

                if (start != DateTime.MinValue)
                {
                    progress =
                        GetProgressPercent(
                            start,
                            OpenPalmStartDelayMs,
                            now) /
                        100f;
                    pinchGesture = false;
                    return true;
                }
            }
            else if (gameState == GameState.Drawing)
            {
                if (waitingForOpenPalmReleaseAfterStart)
                {
                    progress = 1f;
                    pinchGesture = false;
                    return true;
                }

                if (IsPinchDetectionActive())
                {
                    progress =
                        GetPinchProgressPercent(now) /
                        100f;
                    pinchGesture = true;
                    return true;
                }
            }

            progress = 0f;
            pinchGesture = false;
            return false;
        }

        private void ResetPerformanceDiagnostics()
        {
            cameraFrameCount = 0;
            displayFrameCount = 0;
            cameraFramesPerSecond = 0d;
            displayFramesPerSecond = 0d;
            poseFramesPerSecond = 0d;
            latestHandInferenceMilliseconds = 0d;
            latestHandInferenceResult = "WAIT";
            cameraFpsStopwatch.Restart();
            displayFpsStopwatch.Restart();
        }

        private void RecordCameraFrame()
        {
            cameraFrameCount++;

            if (cameraFpsStopwatch.ElapsedMilliseconds <
                1000)
            {
                return;
            }

            cameraFramesPerSecond =
                cameraFrameCount *
                1000d /
                Math.Max(
                    1d,
                    cameraFpsStopwatch
                        .Elapsed.TotalMilliseconds);
            cameraFrameCount = 0;
            cameraFpsStopwatch.Restart();
        }

        private void RecordDisplayFrame()
        {
            displayFrameCount++;

            if (displayFpsStopwatch.ElapsedMilliseconds <
                1000)
            {
                return;
            }

            displayFramesPerSecond =
                displayFrameCount *
                1000d /
                Math.Max(
                    1d,
                    displayFpsStopwatch
                        .Elapsed.TotalMilliseconds);
            displayFrameCount = 0;
            displayFpsStopwatch.Restart();
        }

        private void UpdateAirDrawStatusBar()
        {
            if (lblAirDrawStatus == null ||
                lblAirDrawStatus.IsDisposed)
            {
                return;
            }

            string stateText;

            switch (gameState)
            {
                case GameState.Drawing:
                    stateText = "DRAWING";
                    break;

                case GameState.Scoring:
                    stateText = "SCORING";
                    break;

                case GameState.Finished:
                    stateText = "FINISHED";
                    break;

                default:
                    stateText = "READY";
                    break;
            }

            lblAirDrawStatus.Text =
                stateText +
                "｜" +
                drawingStatusText;

            if (lblSimilarityScore == null ||
                lblSimilarityScore.IsDisposed)
            {
                return;
            }

            if (!lastDrawingScore.HasValue)
            {
                lblSimilarityScore.Text =
                    "形狀相似度：--";
                lblSimilarityScore.ForeColor =
                    Color.DimGray;
                return;
            }

            double score =
                lastDrawingScore.Value;
            lblSimilarityScore.Text =
                "形狀相似度：" +
                score.ToString("0.0") +
                " 分";
            lblSimilarityScore.ForeColor =
                score >= 75d
                    ? Color.Green
                    : score >= 55d
                        ? Color.DarkOrange
                        : Color.Firebrick;
        }

        private List<Detection> PostProcess(Tensor<float> output)
        {
            int numAnchors = 8400;
            List<Detection> detections = new List<Detection>();

            for (int i = 0; i < numAnchors; i++)
            {
                float score = output[0, 4, i];
                if (score < 0.5f) continue;

                Detection det = new Detection
                {
                    X = output[0, 0, i],
                    Y = output[0, 1, i],
                    W = output[0, 2, i],
                    H = output[0, 3, i],
                    Score = score
                };

                for (int k = 0; k < 17; k++)
                {
                    float kx = output[0, 5 + k * 3 + 0, i];
                    float ky = output[0, 5 + k * 3 + 1, i];
                    float ks = output[0, 5 + k * 3 + 2, i];

                    det.Keypoints.Add(new Keypoint { X = kx, Y = ky, Score = ks });
                }

                detections.Add(det);
            }

            return NMS(detections);
        }

        private float IoU(Detection a, Detection b)
        {
            float ax1 = a.X - a.W / 2;
            float ay1 = a.Y - a.H / 2;
            float ax2 = a.X + a.W / 2;
            float ay2 = a.Y + a.H / 2;

            float bx1 = b.X - b.W / 2;
            float by1 = b.Y - b.H / 2;
            float bx2 = b.X + b.W / 2;
            float by2 = b.Y + b.H / 2;

            float interX1 = Math.Max(ax1, bx1);
            float interY1 = Math.Max(ay1, by1);
            float interX2 = Math.Min(ax2, bx2);
            float interY2 = Math.Min(ay2, by2);

            float interArea = Math.Max(0, interX2 - interX1) * Math.Max(0, interY2 - interY1);

            float areaA = (ax2 - ax1) * (ay2 - ay1);
            float areaB = (bx2 - bx1) * (by2 - by1);

            return interArea / (areaA + areaB - interArea);
        }

        private bool IsBodyDetected(Detection person)
        {
            return person != null;
        }

        private bool IsRightHandRaised(Detection person)
        {
            if (person == null || person.Keypoints.Count <= 10)
                return false;

            var wrist = person.Keypoints[10];
            var shoulder = person.Keypoints[6];

            if (wrist.Score < 0.5f || shoulder.Score < 0.5f)
                return false;

            float wristY = (wrist.Y - _padY) / _ratio;
            float shoulderY = (shoulder.Y - _padY) / _ratio;

            return wristY < shoulderY;
        }

        private bool IsRaiseHandConfirmed(bool handRaised)
        {
            if (!handRaised)
            {
                handRaisedStart = DateTime.MinValue;
                return false;
            }

            if (handRaisedStart == DateTime.MinValue)
            {
                handRaisedStart = DateTime.Now;
                return false;
            }

            return (DateTime.Now - handRaisedStart).TotalMilliseconds >= RaiseHandDelayMs;
        }

        private List<Detection> NMS(List<Detection> boxes, float iouThreshold = 0.45f)
        {
            List<Detection> result = new List<Detection>();

            var sorted = boxes.OrderByDescending(b => b.Score).ToList();

            while (sorted.Count > 0)
            {
                var best = sorted[0];
                result.Add(best);
                sorted.RemoveAt(0);

                for (int i = sorted.Count - 1; i >= 0; i--)
                {
                    if (IoU(best, sorted[i]) >= iouThreshold)
                        sorted.RemoveAt(i);
                }
            }

            return result;
        }
    }
}
