using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
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
        YoloV8 yolo;
        InferenceSession yoloSession;
        const int yoloImgWidth = 640;
        const int yoloImgHeight = 640;
        float _ratio;
        int newW;
        int newH;
        int _padX;
        int _padY;

        public class Detection
        {
            public float X;
            public float Y;
            public float W;
            public float H;
            public float Score;
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
            yolo = new YoloV8("yolov8n.onnx");
            string modelPath = "yolov8n.onnx";
            yoloSession = new InferenceSession(modelPath);
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
                IEnum iPixelFormat = nodeMap.GetNode<IEnum>("PixelFormat");
                Console.WriteLine("PixelFormat: " + iPixelFormat.ToString());
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
            Stopwatch sw = new Stopwatch();
            IManagedImageProcessor processor = new ManagedImageProcessor();
            IManagedImage convertedImage = new ManagedImage();
            IManagedImage deepCopyImage = new ManagedImage();
            DenseTensor<float> tensor =
                new DenseTensor<float>(
                    new[] { 1, 3, yoloImgHeight, yoloImgWidth });
            while (grabImage)
            {
                IManagedImage rawImage = cam.GetNextImage();
                processor.Convert(rawImage, convertedImage, PixelFormatEnums.BGR8);
                rawImage.Release();
                deepCopyImage.DeepCopy(convertedImage);
                //DenseTensor<float> tensor = CreateTensorFromFLIR(deepCopyImage);
                CreateTensorFromFLIR(deepCopyImage, ref tensor);
                var output = yoloSession.Run(new[]
                {
                    NamedOnnxValue.CreateFromTensor("images", tensor)
                });
                Tensor<float> resultTensor = output.First().AsTensor<float>();
                //Console.WriteLine(    string.Join(",", resultTensor.Dimensions.ToArray()));
                List<Detection> finalBoxes = PostProcess(resultTensor);
                //Bitmap displayBmp = CreateDisplayImage(deepCopyImage, finalBoxes, ratio, padX, padY);
                Bitmap displayBmp = CreateDisplayImage(deepCopyImage, finalBoxes);
                pBox.BeginInvoke(new InvokeDelegate(pictureBoxInvoke), displayBmp);
            }
            threadComplete = true;
        }

        private void pictureBoxInvoke(Bitmap bmp)
        {
            Image old = pBox.Image;
            pBox.Image = bmp;
            old?.Dispose();
        }

        //private DenseTensor<float> CreateTensorFromFLIR(IManagedImage img)
        private void CreateTensorFromFLIR(IManagedImage img, ref DenseTensor<float> tensor)
        {
            //int srcW = (int)img.Width;
            //int srcH = (int)img.Height;

            //_ratio = Math.Min(
            //    (float)yoloImgWidth / srcW,
            //    (float)yoloImgHeight / srcH);

            //int newW = (int)(srcW * _ratio);
            //int newH = (int)(srcH * _ratio);

            //_padX = (yoloImgWidth - newW) / 2;
            //_padY = (yoloImgHeight - newH) / 2;

            //DenseTensor<float> tensor =
            //    new DenseTensor<float>(
            //        new[] { 1, 3, yoloImgHeight, yoloImgWidth });

            Span<float> span = tensor.Buffer.Span;

            span.Clear(); // padding區域填黑

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

            //return tensor;
        }

        //private Bitmap Letterbox(Bitmap src, int dstSize, out float ratio, out int padX, out int padY)
        private Bitmap Letterbox(Bitmap src)
        {
            //int w = src.Width;
            //int h = src.Height;

            //ratio = Math.Min((float)dstSize / w, (float)dstSize / h);

            //int newW = (int)(w * ratio);
            //int newH = (int)(h * ratio);

            //padX = (dstSize - newW) / 2;
            //padY = (dstSize - newH) / 2;

            //Bitmap dst = new Bitmap(dstSize, dstSize);
            Bitmap dst = new Bitmap(yoloImgWidth, yoloImgHeight);

            using (Graphics g = Graphics.FromImage(dst))
            {
                g.Clear(Color.Black);
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Bilinear;

                //g.DrawImage(src, padX, padY, newW, newH);
                g.DrawImage(src, _padX, _padY, newW, newH);
            }

            return dst;
        }

        private DenseTensor<float> BitmapToTensor(Bitmap bmp)
        {
            DenseTensor<float> tensor =
                new DenseTensor<float>(new[] { 1, 3, 640, 640 });

            var data = bmp.LockBits(
                new Rectangle(0, 0, 640, 640),
                ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb);

            Span<float> span = tensor.Buffer.Span;
            int hw = yoloImgWidth * yoloImgHeight;
            int rOffset = 0;
            int gOffset = hw;
            int bOffset = hw * 2;
            const float INV255 = 1.0f / 255.0f;
            unsafe
            {
                byte* ptr = (byte*)data.Scan0;

                for (int y = 0; y < yoloImgHeight; y++)
                {
                    byte* row = ptr + y * data.Stride;
                    int rowOffset = y * yoloImgWidth;

                    for (int x = 0; x < yoloImgWidth; x++)
                    {
                        int pixelIdx = x * 3;
                        byte b = row[pixelIdx];
                        byte g = row[pixelIdx + 1];
                        byte r = row[pixelIdx + 2];

                        int idx = rowOffset + x;
                        span[rOffset + idx] = r * INV255;
                        span[gOffset + idx] = g * INV255;
                        span[bOffset + idx] = b * INV255;
                    }
                }
            }

            bmp.UnlockBits(data);

            return tensor;
        }

        //private Bitmap CreateDisplayImage(IManagedImage img, List<Detection> boxes, float ratio, float padX, float padY)
        private Bitmap CreateDisplayImage(IManagedImage img, List<Detection> boxes)
        {
            Bitmap bmp = new Bitmap(
                (int)img.Width,
                (int)img.Height,
                (int)img.Stride,
                PixelFormat.Format24bppRgb,
                img.DataPtr);

            Bitmap copy = new Bitmap(bmp);

            using (Graphics g = Graphics.FromImage(copy))
            {
                Pen pen = new Pen(Color.Red, 2);

                foreach (var b in boxes)
                {
                    float x = b.X;
                    float y = b.Y;
                    float w = b.W ;
                    float h = b.H;

                    x -= _padX;
                    y -= _padY;

                    x /= _ratio;
                    y /= _ratio;
                    w /=_ratio;
                    h /= _ratio;

                    float left = x - w / 2;
                    float top = y - h / 2;
                    
                    g.DrawRectangle(pen, left, top, w, h);
                }
            }

            return copy;
        }

        private List<Detection> PostProcess(Tensor<float> output)
        {
            int numAnchors = 8400;
            int numClasses = 80;
            List<Detection> detections = new List<Detection>();
            for (int i = 0; i < numAnchors; i++)
            {
                float personScore = output[0, 4, i];

                if (personScore < 0.5f)
                    continue;

                detections.Add(new Detection
                {
                    X = output[0, 0, i],
                    Y = output[0, 1, i],
                    W = output[0, 2, i],
                    H = output[0, 3, i],
                    Score = personScore
                });
            }
            //for (int i = 0; i < numAnchors; i++)
            //{
            //    float x = output[0, 0, i];
            //    float y = output[0, 1, i];
            //    float w = output[0, 2, i];
            //    float h = output[0, 3, i];

            //    float bestScore = 0;
            //    int classId = -1;

            //    for (int c = 0; c < numClasses; c++)
            //    {
            //        float cls = output[0, 4 + c, i];

            //        if (cls > bestScore)
            //        {
            //            bestScore = cls;
            //            classId = c;
            //        }
            //    }


            //    // COCO: person = 0
            //    if (classId != 0)
            //        continue;

            //    if (bestScore < 0.5f)
            //        continue;

            //    detections.Add(new Detection
            //    {
            //        X = x,
            //        Y = y,
            //        W = w,
            //        H = h,
            //        Score = bestScore
            //    });
            //}
            List<Detection> finalBoxes = NMS(detections);
            return finalBoxes;
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

            float interArea =
                Math.Max(0, interX2 - interX1) *
                Math.Max(0, interY2 - interY1);

            float areaA = (ax2 - ax1) * (ay2 - ay1);
            float areaB = (bx2 - bx1) * (by2 - by1);

            return interArea / (areaA + areaB - interArea);
        }

        private List<Detection> NMS(List<Detection> boxes, float iouThreshold = 0.45f)
        {
            List<Detection> result = new List<Detection>();

            List<Detection> sorted = boxes
                .OrderByDescending(b => b.Score)
                .ToList();

            while (sorted.Count > 0)
            {
                var best = sorted[0];
                result.Add(best);
                sorted.RemoveAt(0);

                for (int i = sorted.Count - 1; i >= 0; i--)
                {
                    if (IoU(best, sorted[i]) >= iouThreshold)
                    {
                        sorted.RemoveAt(i);
                    }
                }
                //sorted = sorted
                //    .Where(b => IoU(best, b) < iouThreshold)
                //    .ToList();
            }

            return result;
        }

        public class YoloV8
        {
            private readonly InferenceSession session;

            public YoloV8(string modelPath)
            {
                session = new InferenceSession(modelPath);
            }

            public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Predict(Bitmap bmp)
            {
                DenseTensor<float> tensor = BitmapToTensor(bmp);

                var inputs = new List<NamedOnnxValue>()
                {
                    NamedOnnxValue.CreateFromTensor(
                    session.InputMetadata.Keys.First(),
                    tensor)
                };

                return session.Run(inputs);
            }
            private DenseTensor<float> BitmapToTensor(Bitmap bmp)
            {
                Bitmap resized = new Bitmap(bmp, new Size(640, 640));

                var tensor = new DenseTensor<float>(new[] { 1, 3, 640, 640 });

                BitmapData data = resized.LockBits(
                    new Rectangle(0, 0, 640, 640),
                    ImageLockMode.ReadOnly,
                    PixelFormat.Format24bppRgb);
                unsafe
                {
                    byte* ptr = (byte*)data.Scan0;

                    for (int y = 0; y < 640; y++)
                    {
                        byte* row = ptr + y * data.Stride;

                        for (int x = 0; x < 640; x++)
                        {
                            int idx = x * 3;

                            byte b = row[idx + 0];
                            byte g = row[idx + 1];
                            byte r = row[idx + 2];

                            tensor[0, 0, y, x] = r / 255f;
                            tensor[0, 1, y, x] = g / 255f;
                            tensor[0, 2, y, x] = b / 255f;
                        }
                    }
                }

                resized.UnlockBits(data);
                return tensor;
            }
        }
    }
}
