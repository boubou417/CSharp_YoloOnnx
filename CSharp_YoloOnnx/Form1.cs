using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading;
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
        DateTime handOnChestStart = DateTime.MinValue;
        const int RaiseHandDelayMs = 800;
        const int HandOnChestDelayMs = 800;

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
        const int StopGestureTailTrimMs = 350;
        const int FingertipMissingBreakMs = 300;
        const int FingertipMarkerVisibleMs = 200;
        const int FingertipInferenceIntervalMs = 67;
        const int IdleHandInferenceIntervalMs = 120;
        const int PinchStartDelayMs = 600;
        const int PinchFinishDelayMs = 600;
        const int PinchTailTrimMs = 250;
        const float MinimumPointDistance = 2f;
        const float PointSmoothingFactor = 0.60f;
        const float FingertipSmoothingFactor = 0.70f;
        const float PinchStartRatio = 0.32f;
        const float PinchReleaseRatio = 0.42f;

        DateTime drawingFinishedAt = DateTime.MinValue;
        DateTime fingertipMissingSince = DateTime.MinValue;
        DateTime lastFingertipSeenAt = DateTime.MinValue;
        DateTime lastFingertipInferenceAt = DateTime.MinValue;
        DateTime lastIdleHandInferenceAt = DateTime.MinValue;
        DateTime leftStartPinchAt = DateTime.MinValue;
        DateTime rightStartPinchAt = DateTime.MinValue;
        DateTime pinchStartedAt = DateTime.MinValue;
        string drawingStatusText =
            "左右手皆可｜食指與拇指捏合 0.6 秒開始";
        string templateImagePath = string.Empty;
        double? lastDrawingScore;
        Button btnSelectTemplate;
        FingertipTracker fingertipTracker;
        PointF? lastFingertipPoint;
        PointF? displayedFingertipPoint;
        PointF? displayedThumbPoint;
        float displayedFingertipPresence;
        float latestPinchRatio = float.MaxValue;
        bool drawingStrokeStartPending = true;
        bool pinchInProgress;
        bool waitingForPinchReleaseAfterStart;
        DrawingHand activeDrawingHand = DrawingHand.None;
        DrawingHand displayedDrawingHand = DrawingHand.None;
        Bitmap pendingDisplayImage;
        int displayUpdateScheduled;

        InferenceSession yoloSession;

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

        public Form1()
        {
            InitializeComponent();

            Text = "CSharp YOLO ONNX V1.1";
            panelToolBar.Dock = DockStyle.Top;
            panelToolBar.Height = 40;
            panelStatusBar.Dock = DockStyle.Bottom;
            panelStatusBar.Height = 20;
            panelImage.Dock = DockStyle.Fill;
            pBox.Dock = DockStyle.Fill;
            pBox.SizeMode = PictureBoxSizeMode.Zoom;
            InitializeAirDrawControls();
            TryLoadDefaultTemplate();
            InitializeFingertipTracker();
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
            yoloSession = new InferenceSession(modelPath);
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
            DisposePendingDisplayImage();

            if (fingertipTracker != null)
            {
                fingertipTracker.Dispose();
                fingertipTracker = null;
            }
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
                "｜左右手捏合開始";
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
                    "｜左右手捏合開始";
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
                Thread thread = new Thread(ThreadGetImages);
                thread.Start();

                streaming = true;
                btnGrab.Text = "Stop";
            }
            else
            {
                grabImage = false;

                while (!threadComplete)
                    Thread.Sleep(200);

                cam.EndAcquisition();

                streaming = false;
                threadComplete = false;
                btnGrab.Text = "Grab";
            }
        }

        public delegate void InvokeDelegate(Bitmap bmp);

        private void ThreadGetImages()
        {
            IManagedImageProcessor processor = new ManagedImageProcessor();
            IManagedImage convertedImage = new ManagedImage();
            IManagedImage deepCopyImage = new ManagedImage();

            DenseTensor<float> tensor = new DenseTensor<float>(new[] { 1, 3, yoloImgHeight, yoloImgWidth });

            while (grabImage)
            {
                IManagedImage rawImage = cam.GetNextImage();
                processor.Convert(rawImage, convertedImage, PixelFormatEnums.BGR8);
                rawImage.Release();

                deepCopyImage.DeepCopy(convertedImage);

                CreateTensorFromFLIR(deepCopyImage, ref tensor);

                List<Detection> finalBoxes;

                using (var output = yoloSession.Run(new[]
                {
                    NamedOnnxValue.CreateFromTensor("images", tensor)
                }))
                {
                    Tensor<float> resultTensor =
                        output.First().AsTensor<float>();
                    finalBoxes = PostProcess(resultTensor);
                }

                // 🔥 揮手偵測（主角：最大人）
                isWaving = false;
                var main = finalBoxes.OrderByDescending(d => d.W * d.H).FirstOrDefault();

                //if (main != null)
                //{
                //    bool handRaised = IsRightHandRaised(main);
                //    bool handOnChest = IsRightHandOnChest(main);
                //    var wrist = main.Keypoints[10]; // 右手腕
                //    var shoulder = main.Keypoints[6]; // 右肩

                //    if (wrist.Score > 0.5f && shoulder.Score > 0.5f)
                //    {
                //        float x = (wrist.X - _padX) / _ratio;
                //        float y = (wrist.Y - _padY) / _ratio;

                //        //if (y < (shoulder.Y - _padY) / _ratio)
                //        if (handRaised)
                //        {
                //            // 第一次舉手，開始收集
                //            if (gameState == GameState.Idle)
                //            {
                //                StartDrawing();
                //            }

                //            PointF pt = new PointF(x, y);

                //            rightHandHistory.Enqueue(x);

                //            handTrail.Add(pt);
                //            AddDrawingPoint(pt);

                //            if (rightHandHistory.Count > 12)
                //                rightHandHistory.Dequeue();

                //            if (handTrail.Count > 20)
                //                handTrail.RemoveAt(0);

                //            if (rightHandHistory.Count >= 10)
                //            {
                //                float minX = rightHandHistory.Min();
                //                float maxX = rightHandHistory.Max();

                //                if ((maxX - minX) > 80)
                //                    isWaving = true;
                //            }
                //        }
                //        else
                //        {
                //            // 手放下
                //            if (gameState == GameState.Drawing)
                //            {
                //                StopDrawing();
                //            }

                //            rightHandHistory.Clear();
                //            handTrail.Clear();
                //        }
                //        //if (y < (shoulder.Y - _padY) / _ratio)
                //        //{
                //        //    rightHandHistory.Enqueue(x);
                //        //    handTrail.Add(new PointF(x, y));

                //        //    if (rightHandHistory.Count > 12) rightHandHistory.Dequeue();
                //        //    if (handTrail.Count > 20) handTrail.RemoveAt(0);

                //        //    if (rightHandHistory.Count >= 10)
                //        //    {
                //        //        float minX = rightHandHistory.Min();
                //        //        float maxX = rightHandHistory.Max();
                //        //        if ((maxX - minX) > 80) isWaving = true;
                //        //    }
                //        //}
                //        //else
                //        //{
                //        //    rightHandHistory.Clear();
                //        //    handTrail.Clear();
                //        //}
                //    }
                //}

                Bitmap displayBmp = CreateDisplayImage(
                    deepCopyImage,
                    finalBoxes,
                    main);
                QueueDisplayImage(displayBmp);
            }

            threadComplete = true;
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
            handOnChestStart = DateTime.MinValue;
            drawingFinishedAt = DateTime.MinValue;
            fingertipMissingSince = DateTime.MinValue;
            lastFingertipSeenAt = DateTime.MinValue;
            lastFingertipInferenceAt = DateTime.MinValue;
            lastFingertipPoint = null;
            displayedFingertipPoint = null;
            displayedThumbPoint = null;
            displayedFingertipPresence = 0f;
            ResetPinchGesture(false);
            ResetStartPinchGestures();
            drawingStrokeStartPending = true;
            lastDrawingScore = null;
            activeDrawingHand = hand;
            displayedDrawingHand = hand;
            waitingForPinchReleaseAfterStart = true;

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
                lastFingertipSeenAt = DateTime.Now;
            }

            gameState = GameState.Drawing;
            drawingStatusText =
                GetHandDisplayName(hand) +
                "開始成功｜請先放開捏合";

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
            waitingForPinchReleaseAfterStart = false;
            rightHandHistory.Clear();
            isWaving = false;
            handRaisedStart = DateTime.MinValue;
            handOnChestStart = DateTime.MinValue;
            displayedFingertipPoint = null;
            displayedThumbPoint = null;
            displayedDrawingHand = DrawingHand.None;
            activeDrawingHand = DrawingHand.None;

            Debug.WriteLine("===== Finish Drawing =====");
            Debug.WriteLine($"Trajectory Points = {drawingPoints.Count}");

            if (drawingPoints.Count < MinimumDrawingPoints)
            {
                drawingStatusText =
                    "軌跡太短，未儲存｜請重新捏合開始";
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
                    ResetStartPinchGestures();
                    activeDrawingHand = DrawingHand.None;
                    displayedDrawingHand = DrawingHand.None;
                    waitingForPinchReleaseAfterStart = false;
                    isWaving = false;
                    drawingStatusText =
                        string.IsNullOrWhiteSpace(templateImagePath)
                            ? GetIdleInstruction()
                            : "比對圖：" +
                              Path.GetFileName(templateImagePath) +
                              "｜左右手捏合開始";
                }

                return;
            }

            if (gameState == GameState.Scoring)
                return;

            if (main == null || main.Keypoints.Count <= 10 || _ratio <= 0f)
            {
                handRaisedStart = DateTime.MinValue;
                handOnChestStart = DateTime.MinValue;

                if (gameState == GameState.Idle)
                {
                    ResetStartPinchGestures();
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
                    bool rawHandOnChest = IsHandOnChest(
                        main,
                        activeDrawingHand);
                    bool handOnChest =
                        IsHandOnChestConfirmed(rawHandOnChest);

                    if (!rawHandOnChest)
                    {
                        PointF fingertip;

                        if (TryGetFingertipPoint(
                            frame,
                            main,
                            activeDrawingHand,
                            out fingertip))
                        {
                            if (waitingForPinchReleaseAfterStart)
                            {
                                if (latestPinchRatio >=
                                    PinchReleaseRatio)
                                {
                                    waitingForPinchReleaseAfterStart =
                                        false;
                                    ResetPinchGesture(false);
                                    drawingStrokeStartPending = true;
                                    lastFingertipPoint = null;
                                    drawingStatusText =
                                        GetHandDisplayName(
                                            activeDrawingHand) +
                                        "已放開｜開始用食指畫圖";
                                }
                                else
                                {
                                    drawingStatusText =
                                        GetHandDisplayName(
                                            activeDrawingHand) +
                                        "開始成功｜請先放開捏合";
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
                            else if (pinchInProgress)
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
                                "食指｜請讓手腕與手肘留在畫面";
                        }
                    }
                    else
                    {
                        ResetPinchGesture(true);
                        MarkFingertipMissing();
                    }

                    if (handOnChest)
                    {
                        StopDrawing(
                            handOnChestStart,
                            StopGestureTailTrimMs);
                    }

                    break;

                case GameState.Finished:
                    break;

                case GameState.Countdown:

                    break;

                case GameState.Scoring:

                    break;
            }
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

            bool leftConfirmed = UpdateStartPinchCandidate(
                DrawingHand.Left,
                leftDetected ? leftResult : null,
                now);
            bool rightConfirmed = UpdateStartPinchCandidate(
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

            if (leftStartPinchAt != DateTime.MinValue &&
                leftDetected)
            {
                displayResult = leftResult;
                displayHand = DrawingHand.Left;
            }
            else if (rightStartPinchAt != DateTime.MinValue &&
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

            DateTime activeStart = GetStartPinchAt(displayHand);

            if (activeStart != DateTime.MinValue)
            {
                int progress = GetProgressPercent(
                    activeStart,
                    PinchStartDelayMs,
                    now);
                drawingStatusText =
                    GetHandDisplayName(displayHand) +
                    "捏合開始 " +
                    progress +
                    "%";
            }
            else
            {
                drawingStatusText = GetIdleInstruction();
            }
        }

        private bool UpdateStartPinchCandidate(
            DrawingHand hand,
            FingertipResult result,
            DateTime now)
        {
            DateTime startedAt = GetStartPinchAt(hand);

            if (result == null)
            {
                SetStartPinchAt(hand, DateTime.MinValue);
                return false;
            }

            if (result.PinchRatio <= PinchStartRatio)
            {
                if (startedAt == DateTime.MinValue)
                {
                    startedAt = now;
                    SetStartPinchAt(hand, startedAt);
                }

                return
                    (now - startedAt).TotalMilliseconds >=
                    PinchStartDelayMs;
            }

            if (result.PinchRatio >= PinchReleaseRatio)
                SetStartPinchAt(hand, DateTime.MinValue);

            return false;
        }

        private void ResetStartPinchGestures()
        {
            leftStartPinchAt = DateTime.MinValue;
            rightStartPinchAt = DateTime.MinValue;
        }

        private DateTime GetStartPinchAt(DrawingHand hand)
        {
            if (hand == DrawingHand.Left)
                return leftStartPinchAt;

            if (hand == DrawingHand.Right)
                return rightStartPinchAt;

            return DateTime.MinValue;
        }

        private void SetStartPinchAt(
            DrawingHand hand,
            DateTime value)
        {
            if (hand == DrawingHand.Left)
                leftStartPinchAt = value;
            else if (hand == DrawingHand.Right)
                rightStartPinchAt = value;
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
            return "左右手皆可｜食指與拇指捏合 0.6 秒開始";
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
            lastFingertipSeenAt = DateTime.Now;
        }

        private void ClearDisplayedHand()
        {
            displayedDrawingHand = DrawingHand.None;
            displayedFingertipPoint = null;
            displayedThumbPoint = null;
            displayedFingertipPresence = 0f;
            latestPinchRatio = float.MaxValue;
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

            if (elbowKeypoint.Score < 0.5f ||
                wristKeypoint.Score < 0.5f)
            {
                return false;
            }

            PointF elbow = KeypointToImagePoint(elbowKeypoint);
            PointF wrist = KeypointToImagePoint(wristKeypoint);

            try
            {
                return fingertipTracker.TryDetect(
                    frame,
                    wrist,
                    elbow,
                    out result);
            }
            catch (Exception ex)
            {
                Debug.WriteLine(
                    "Hand inference error (" +
                    GetHandDisplayName(hand) +
                    "): " +
                    ex);
                return false;
            }
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

            if (lastFingertipPoint.HasValue &&
                lastFingertipSeenAt != DateTime.MinValue &&
                (now - lastFingertipSeenAt).TotalMilliseconds <
                    FingertipMissingBreakMs)
            {
                PointF previous = lastFingertipPoint.Value;
                float dx = detected.X - previous.X;
                float dy = detected.Y - previous.Y;
                float maximumJump = GetMaximumFingertipJump(person);

                if (dx * dx + dy * dy > maximumJump * maximumJump)
                    return false;

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
                if (latestPinchRatio > PinchStartRatio)
                    return false;

                pinchInProgress = true;
                pinchStartedAt = now;
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
            if (!pinchInProgress ||
                pinchStartedAt == DateTime.MinValue)
            {
                return 0;
            }

            double progress =
                (now - pinchStartedAt).TotalMilliseconds /
                PinchFinishDelayMs;

            return (int)Math.Max(
                0d,
                Math.Min(100d, progress * 100d));
        }

        private void ResetPinchGesture(bool startNewStroke)
        {
            bool wasPinching = pinchInProgress;

            pinchInProgress = false;
            pinchStartedAt = DateTime.MinValue;
            latestPinchRatio = float.MaxValue;

            if (startNewStroke && wasPinching)
            {
                drawingStrokeStartPending = true;
                lastFingertipPoint = null;
            }
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
            pBox.Refresh();
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
                    pictureBoxInvoke(latest);
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

        private Bitmap CreateDisplayImage(IManagedImage img, List<Detection> boxes, Detection main)
        {
            Bitmap copy = CopyManagedImageToBitmap(img);

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
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;

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
                DrawAirDrawStatus(g, copy.Width);
                DrawLiveDiagnostics(g, boxes.Count);
            }

            return copy;
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

        private void DrawLiveDiagnostics(
            Graphics graphics,
            int personCount)
        {
            string text = "LIVE | YOLO persons: " + personCount;

            using (Font font = new Font(
                "Microsoft JhengHei UI",
                10f,
                FontStyle.Bold))
            using (Brush background =
                new SolidBrush(Color.FromArgb(170, 0, 0, 0)))
            {
                SizeF textSize = graphics.MeasureString(text, font);
                RectangleF backgroundBounds = new RectangleF(
                    8f,
                    8f,
                    textSize.Width + 16f,
                    textSize.Height + 8f);

                graphics.FillRectangle(background, backgroundBounds);
                graphics.DrawString(
                    text,
                    font,
                    Brushes.Lime,
                    16f,
                    12f);
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
            float pinchProgress;
            bool showPinchProgress =
                TryGetDisplayedPinchProgress(out pinchProgress);

            using (Pen outline = new Pen(Color.Cyan, 3f))
            using (Pen thumbOutline = new Pen(Color.Magenta, 3f))
            using (Pen pinchLine = new Pen(Color.Orange, 2f))
            using (Pen progressPen = new Pen(Color.Orange, 4f))
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

                    if (showPinchProgress)
                    {
                        graphics.DrawLine(
                            pinchLine,
                            thumb,
                            point);

                        graphics.DrawArc(
                            progressPen,
                            point.X - 14f,
                            point.Y - 14f,
                            28f,
                            28f,
                            -90f,
                            360f * pinchProgress);
                    }
                }

                if (latestPinchRatio < float.MaxValue)
                {
                    graphics.DrawString(
                        "捏合 " +
                        latestPinchRatio.ToString("0.00"),
                        font,
                        showPinchProgress
                            ? Brushes.Orange
                            : Brushes.Magenta,
                        point.X + 12f,
                        point.Y + 6f);
                }
            }
        }

        private bool TryGetDisplayedPinchProgress(
            out float progress)
        {
            DateTime now = DateTime.Now;

            if (gameState == GameState.Idle)
            {
                DateTime start =
                    GetStartPinchAt(displayedDrawingHand);

                if (start != DateTime.MinValue)
                {
                    progress =
                        GetProgressPercent(
                            start,
                            PinchStartDelayMs,
                            now) /
                        100f;
                    return true;
                }
            }
            else if (gameState == GameState.Drawing)
            {
                if (waitingForPinchReleaseAfterStart)
                {
                    progress = 1f;
                    return true;
                }

                if (pinchInProgress)
                {
                    progress =
                        GetPinchProgressPercent(now) /
                        100f;
                    return true;
                }
            }

            progress = 0f;
            return false;
        }

        private void DrawAirDrawStatus(Graphics graphics, int imageWidth)
        {
            string stateText;
            Color stateColor;

            switch (gameState)
            {
                case GameState.Drawing:
                    stateText = "DRAWING";
                    stateColor = Color.Lime;
                    break;

                case GameState.Scoring:
                    stateText = "SCORING";
                    stateColor = Color.Orange;
                    break;

                case GameState.Finished:
                    stateText = "FINISHED";
                    stateColor = lastDrawingScore.HasValue ? Color.Cyan : Color.Yellow;
                    break;

                default:
                    stateText = "READY";
                    stateColor = Color.White;
                    break;
            }

            string message = stateText + "｜" + drawingStatusText;

            using (Font font = new Font("Microsoft JhengHei UI", 14f, FontStyle.Bold))
            {
                SizeF textSize = graphics.MeasureString(message, font);
                float width = Math.Min(imageWidth - 20f, textSize.Width + 24f);

                using (Brush background = new SolidBrush(Color.FromArgb(170, 0, 0, 0)))
                using (Brush foreground = new SolidBrush(stateColor))
                {
                    graphics.FillRectangle(background, 10f, 10f, width, textSize.Height + 16f);
                    graphics.DrawString(message, font, foreground, 20f, 18f);
                }
            }
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

        private bool IsHandOnChest(
            Detection person,
            DrawingHand hand)
        {
            if (person == null ||
                person.Keypoints.Count <= 10 ||
                hand == DrawingHand.None)
                return false;

            int wristIndex =
                hand == DrawingHand.Left ? 9 : 10;
            var wrist = person.Keypoints[wristIndex];
            var leftShoulder = person.Keypoints[5];
            var rightShoulder = person.Keypoints[6];

            if (wrist.Score < 0.5f)
                return false;

            if (leftShoulder.Score < 0.5f)
                return false;

            if (rightShoulder.Score < 0.5f)
                return false;

            float wristX = (wrist.X - _padX) / _ratio;
            float wristY = (wrist.Y - _padY) / _ratio;

            float leftShoulderX = (leftShoulder.X - _padX) / _ratio;
            float leftShoulderY = (leftShoulder.Y - _padY) / _ratio;
            float rightShoulderX = (rightShoulder.X - _padX) / _ratio;
            float rightShoulderY = (rightShoulder.Y - _padY) / _ratio;

            float shoulderMidX = (leftShoulderX + rightShoulderX) / 2f;
            float shoulderMidY = (leftShoulderY + rightShoulderY) / 2f;
            float shoulderDx = rightShoulderX - leftShoulderX;
            float shoulderDy = rightShoulderY - leftShoulderY;
            float shoulderWidth = (float)Math.Sqrt(
                shoulderDx * shoulderDx + shoulderDy * shoulderDy);

            if (shoulderWidth < 20f)
                return false;

            // 胸前位置會隨人物在畫面中的大小調整，避免固定像素門檻造成遠近差異。
            float chestX = shoulderMidX;
            float chestY = shoulderMidY + shoulderWidth * 0.45f;
            float radiusX = Math.Max(35f, shoulderWidth * 0.65f);
            float radiusY = Math.Max(35f, shoulderWidth * 0.55f);

            double dx = wristX - chestX;
            double dy = wristY - chestY;

            double normalizedDistance =
                dx * dx / (radiusX * radiusX) +
                dy * dy / (radiusY * radiusY);

            return normalizedDistance <= 1d;
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

        private bool IsHandOnChestConfirmed(bool handOnChest)
        {
            if (!handOnChest)
            {
                handOnChestStart = DateTime.MinValue;
                return false;
            }

            if (handOnChestStart == DateTime.MinValue)
            {
                handOnChestStart = DateTime.Now;
                return false;
            }

            return (DateTime.Now - handOnChestStart).TotalMilliseconds >= HandOnChestDelayMs;
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
