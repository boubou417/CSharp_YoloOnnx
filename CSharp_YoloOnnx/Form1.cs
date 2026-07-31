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
        const float MinimumPointDistance = 2f;
        const float PointSmoothingFactor = 0.60f;
        const float FingertipSmoothingFactor = 0.70f;

        DateTime drawingFinishedAt = DateTime.MinValue;
        DateTime fingertipMissingSince = DateTime.MinValue;
        DateTime lastFingertipSeenAt = DateTime.MinValue;
        DateTime lastFingertipInferenceAt = DateTime.MinValue;
        string drawingStatusText = "舉起右手並停留 0.8 秒開始畫圖";
        string templateImagePath = string.Empty;
        double? lastDrawingScore;
        Button btnSelectTemplate;
        FingertipTracker fingertipTracker;
        PointF? lastFingertipPoint;
        PointF? displayedFingertipPoint;
        float displayedFingertipPresence;
        bool drawingStrokeStartPending = true;
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

        public Form1()
        {
            InitializeComponent();

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
            drawingStatusText = "比對圖：" + Path.GetFileName(templateImagePath) + "｜舉起右手開始";
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
                    "｜舉起右手開始";
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

        private void StartDrawing()
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
            displayedFingertipPresence = 0f;
            drawingStrokeStartPending = true;
            lastDrawingScore = null;

            gameState = GameState.Drawing;
            drawingStatusText =
                "繪圖中（右手食指）｜手放回胸前並停留 0.8 秒完成";

            Debug.WriteLine("===== Start Drawing =====");
        }

        private void StopDrawing()
        {
            if (gameState != GameState.Drawing)
                return;

            TrimStopGestureTail();
            rightHandHistory.Clear();
            isWaving = false;
            handRaisedStart = DateTime.MinValue;
            handOnChestStart = DateTime.MinValue;
            displayedFingertipPoint = null;

            Debug.WriteLine("===== Finish Drawing =====");
            Debug.WriteLine($"Trajectory Points = {drawingPoints.Count}");

            if (drawingPoints.Count < MinimumDrawingPoints)
            {
                drawingStatusText = "軌跡太短，未儲存｜請重新舉手畫圖";
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

        private void TrimStopGestureTail()
        {
            if (drawingPoints.Count == 0 ||
                drawingPointTimes.Count != drawingPoints.Count)
                return;

            DateTime chestDetectedAt =
                handOnChestStart == DateTime.MinValue
                    ? DateTime.Now
                    : handOnChestStart;

            DateTime cutoff = chestDetectedAt.AddMilliseconds(-StopGestureTailTrimMs);
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
                    displayedFingertipPresence = 0f;
                    fingertipMissingSince = DateTime.MinValue;
                    lastFingertipSeenAt = DateTime.MinValue;
                    lastFingertipInferenceAt = DateTime.MinValue;
                    drawingStrokeStartPending = true;
                    isWaving = false;
                    drawingStatusText =
                        string.IsNullOrWhiteSpace(templateImagePath)
                            ? "舉起右手並停留 0.8 秒開始畫圖"
                            : "比對圖：" +
                              Path.GetFileName(templateImagePath) +
                              "｜舉起右手開始";
                }

                return;
            }

            if (gameState == GameState.Scoring)
                return;

            if (main == null || main.Keypoints.Count <= 10 || _ratio <= 0f)
            {
                handRaisedStart = DateTime.MinValue;
                handOnChestStart = DateTime.MinValue;
                MarkFingertipMissing();
                return;
            }

            bool rawHandRaised = IsRightHandRaised(main);
            bool rawHandOnChest = IsRightHandOnChest(main);

            bool handRaised =
                IsRaiseHandConfirmed(rawHandRaised);

            bool handOnChest =
                gameState == GameState.Drawing &&
                IsHandOnChestConfirmed(rawHandOnChest);

            var wrist = main.Keypoints[10];

            if (wrist.Score < 0.5f)
            {
                MarkFingertipMissing();
                return;
            }

            switch (gameState)
            {
                case GameState.Idle:

                    if (handRaised)
                    {
                        if (fingertipTracker == null)
                        {
                            handRaisedStart = DateTime.MinValue;
                            drawingStatusText =
                                "指尖模型未載入｜請確認模型與 TensorFlow Lite 套件";
                        }
                        else
                        {
                            StartDrawing();
                        }
                    }

                    break;

                case GameState.Drawing:

                    if (!rawHandOnChest)
                    {
                        PointF fingertip;

                        if (TryGetFingertipPoint(frame, main, out fingertip))
                        {
                            AddDrawingPoint(fingertip);
                            drawingStatusText =
                                "繪圖中（右手食指）｜手放回胸前並停留 0.8 秒完成";
                        }
                        else
                        {
                            MarkFingertipMissing();
                            drawingStatusText =
                                "正在尋找右手食指｜請伸直食指並面向相機";
                        }
                    }
                    else
                    {
                        MarkFingertipMissing();
                    }

                    if (handOnChest)
                    {
                        StopDrawing();
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

        private bool TryGetFingertipPoint(
            Bitmap frame,
            Detection person,
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

            Keypoint elbowKeypoint = person.Keypoints[8];
            Keypoint wristKeypoint = person.Keypoints[10];

            if (elbowKeypoint.Score < 0.5f ||
                wristKeypoint.Score < 0.5f)
            {
                return false;
            }

            PointF elbow = KeypointToImagePoint(elbowKeypoint);
            PointF wrist = KeypointToImagePoint(wristKeypoint);
            FingertipResult result;

            try
            {
                if (!fingertipTracker.TryDetect(
                    frame,
                    wrist,
                    elbow,
                    out result))
                {
                    return false;
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine("Fingertip inference error: " + ex);
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
            displayedFingertipPoint = detected;
            displayedFingertipPresence = result.HandPresence;
            lastFingertipSeenAt = now;
            fingertipMissingSince = DateTime.MinValue;
            fingertip = detected;

            return true;
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
                displayedFingertipPoint = null;
                displayedFingertipPresence = 0f;
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
            Bitmap copy;

            using (Bitmap bmp = new Bitmap(
                (int)img.Width,
                (int)img.Height,
                (int)img.Stride,
                PixelFormat.Format24bppRgb,
                img.DataPtr))
            {
                copy = bmp.Clone(
                    new Rectangle(0, 0, bmp.Width, bmp.Height),
                    PixelFormat.Format24bppRgb);
            }

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
            if (gameState != GameState.Drawing ||
                !displayedFingertipPoint.HasValue)
            {
                return;
            }

            PointF point = displayedFingertipPoint.Value;

            using (Pen outline = new Pen(Color.Cyan, 3f))
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
                    "食指 " +
                    (displayedFingertipPresence * 100f).ToString("0") +
                    "%",
                    font,
                    Brushes.Cyan,
                    point.X + 12f,
                    point.Y - 12f);
            }
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

        private bool IsRightHandOnChest(Detection person)
        {
            if (person == null || person.Keypoints.Count <= 10)
                return false;

            var wrist = person.Keypoints[10];
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

        private bool IsDrawingFinished(Detection person)
        {
            return IsRightHandOnChest(person);
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
