using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Text;

namespace CSharp_YoloOnnx
{
    public sealed class AirDrawSaveResult
    {
        public DateTime CreatedAt { get; set; }
        public string ImagePath { get; set; }
        public string JsonPath { get; set; }
        public int CanvasWidth { get; set; }
        public int CanvasHeight { get; set; }
        public List<PointF> SourcePoints { get; set; }
        public List<PointF> CanvasPoints { get; set; }
        public List<int> StrokeStartIndices { get; set; }
    }

    public static class AirDrawStorage
    {
        private const int DefaultCanvasSize = 512;
        private const int CanvasPadding = 32;

        public static AirDrawSaveResult Save(IList<PointF> points, string outputDirectory)
        {
            return Save(points, null, outputDirectory);
        }

        public static AirDrawSaveResult Save(
            IList<PointF> points,
            IList<int> strokeStartIndices,
            string outputDirectory)
        {
            if (points == null)
                throw new ArgumentNullException(nameof(points));

            if (points.Count < 2)
                throw new ArgumentException("At least two drawing points are required.", nameof(points));

            Directory.CreateDirectory(outputDirectory);

            DateTime createdAt = DateTime.Now;
            string fileStem = "Drawing_" + createdAt.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
            string imagePath = Path.Combine(outputDirectory, fileStem + ".png");
            string jsonPath = Path.Combine(outputDirectory, fileStem + ".json");

            List<PointF> sourcePoints = new List<PointF>(points.Count);
            for (int i = 0; i < points.Count; i++)
                sourcePoints.Add(points[i]);

            List<PointF> canvasPoints = NormalizeForCanvas(
                sourcePoints,
                DefaultCanvasSize,
                DefaultCanvasSize,
                CanvasPadding);

            List<int> normalizedStrokeStarts = NormalizeStrokeStarts(
                strokeStartIndices,
                sourcePoints.Count);

            RenderDrawing(
                canvasPoints,
                normalizedStrokeStarts,
                imagePath,
                DefaultCanvasSize,
                DefaultCanvasSize);

            AirDrawSaveResult result = new AirDrawSaveResult
            {
                CreatedAt = createdAt,
                ImagePath = imagePath,
                JsonPath = jsonPath,
                CanvasWidth = DefaultCanvasSize,
                CanvasHeight = DefaultCanvasSize,
                SourcePoints = sourcePoints,
                CanvasPoints = canvasPoints,
                StrokeStartIndices = normalizedStrokeStarts
            };

            WriteMetadata(result, null, null);
            return result;
        }

        public static void WriteMetadata(
            AirDrawSaveResult result,
            string templateImagePath,
            double? similarityScore)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            using (StreamWriter writer = new StreamWriter(
                result.JsonPath,
                false,
                new UTF8Encoding(false)))
            {
                writer.WriteLine("{");
                writer.WriteLine("  \"version\": 2,");
                writer.WriteLine(
                    "  \"createdAt\": \"" +
                    result.CreatedAt.ToString("o", CultureInfo.InvariantCulture) +
                    "\",");
                writer.WriteLine("  \"imageFile\": \"" + EscapeJson(Path.GetFileName(result.ImagePath)) + "\",");
                writer.WriteLine("  \"canvasWidth\": " + result.CanvasWidth + ",");
                writer.WriteLine("  \"canvasHeight\": " + result.CanvasHeight + ",");
                writer.WriteLine("  \"sourcePointCount\": " + result.SourcePoints.Count + ",");
                writer.WriteLine(
                    "  \"strokeCount\": " +
                    result.StrokeStartIndices.Count +
                    ",");

                WritePointArray(writer, "sourcePoints", result.SourcePoints, 1f, 1f, true);
                WritePointArray(
                    writer,
                    "normalizedPoints",
                    result.CanvasPoints,
                    result.CanvasWidth,
                    result.CanvasHeight,
                    true);
                WriteIntegerArray(
                    writer,
                    "strokeStartIndices",
                    result.StrokeStartIndices,
                    true);

                if (!string.IsNullOrWhiteSpace(templateImagePath) && similarityScore.HasValue)
                {
                    writer.WriteLine("  \"comparison\": {");
                    writer.WriteLine(
                        "    \"templateFile\": \"" +
                        EscapeJson(Path.GetFileName(templateImagePath)) +
                        "\",");
                    writer.WriteLine(
                        "    \"similarityScore\": " +
                        similarityScore.Value.ToString("0.##", CultureInfo.InvariantCulture));
                    writer.WriteLine("  }");
                }
                else
                {
                    writer.WriteLine("  \"comparison\": null");
                }

                writer.WriteLine("}");
            }
        }

        private static List<PointF> NormalizeForCanvas(
            IList<PointF> points,
            int canvasWidth,
            int canvasHeight,
            int padding)
        {
            float minX = float.MaxValue;
            float minY = float.MaxValue;
            float maxX = float.MinValue;
            float maxY = float.MinValue;

            for (int i = 0; i < points.Count; i++)
            {
                minX = Math.Min(minX, points[i].X);
                minY = Math.Min(minY, points[i].Y);
                maxX = Math.Max(maxX, points[i].X);
                maxY = Math.Max(maxY, points[i].Y);
            }

            float rangeX = Math.Max(0f, maxX - minX);
            float rangeY = Math.Max(0f, maxY - minY);
            float availableWidth = Math.Max(1f, canvasWidth - padding * 2f);
            float availableHeight = Math.Max(1f, canvasHeight - padding * 2f);

            float scaleX = rangeX > 0.001f ? availableWidth / rangeX : float.MaxValue;
            float scaleY = rangeY > 0.001f ? availableHeight / rangeY : float.MaxValue;
            float scale = Math.Min(scaleX, scaleY);

            if (scale == float.MaxValue || float.IsInfinity(scale) || float.IsNaN(scale))
                scale = 1f;

            float drawingWidth = rangeX * scale;
            float drawingHeight = rangeY * scale;
            float offsetX = (canvasWidth - drawingWidth) / 2f - minX * scale;
            float offsetY = (canvasHeight - drawingHeight) / 2f - minY * scale;

            List<PointF> normalized = new List<PointF>(points.Count);
            for (int i = 0; i < points.Count; i++)
            {
                normalized.Add(new PointF(
                    points[i].X * scale + offsetX,
                    points[i].Y * scale + offsetY));
            }

            return normalized;
        }

        private static void RenderDrawing(
            IList<PointF> points,
            IList<int> strokeStartIndices,
            string imagePath,
            int width,
            int height)
        {
            using (Bitmap bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb))
            using (Graphics graphics = Graphics.FromImage(bitmap))
            using (Pen pen = new Pen(Color.White, 8f))
            {
                graphics.Clear(Color.Black);
                graphics.SmoothingMode = SmoothingMode.AntiAlias;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                pen.StartCap = LineCap.Round;
                pen.EndCap = LineCap.Round;
                pen.LineJoin = LineJoin.Round;

                for (int strokeIndex = 0;
                    strokeIndex < strokeStartIndices.Count;
                    strokeIndex++)
                {
                    int start = strokeStartIndices[strokeIndex];
                    int end =
                        strokeIndex + 1 < strokeStartIndices.Count
                            ? strokeStartIndices[strokeIndex + 1]
                            : points.Count;
                    int count = end - start;

                    if (count <= 0)
                        continue;

                    if (count == 1)
                    {
                        graphics.FillEllipse(
                            Brushes.White,
                            points[start].X - 4f,
                            points[start].Y - 4f,
                            8f,
                            8f);
                    }
                    else
                    {
                        graphics.DrawLines(
                            pen,
                            ToPointArray(points, start, count));
                    }
                }

                bitmap.Save(imagePath, ImageFormat.Png);
            }
        }

        private static PointF[] ToPointArray(
            IList<PointF> points,
            int start,
            int count)
        {
            PointF[] result = new PointF[count];
            for (int i = 0; i < count; i++)
                result[i] = points[start + i];

            return result;
        }

        private static List<int> NormalizeStrokeStarts(
            IList<int> strokeStartIndices,
            int pointCount)
        {
            List<int> result = new List<int>();

            if (pointCount <= 0)
                return result;

            result.Add(0);

            if (strokeStartIndices == null)
                return result;

            for (int i = 0; i < strokeStartIndices.Count; i++)
            {
                int value = strokeStartIndices[i];

                if (value <= 0 || value >= pointCount)
                    continue;

                if (!result.Contains(value))
                    result.Add(value);
            }

            result.Sort();
            return result;
        }

        private static void WritePointArray(
            StreamWriter writer,
            string name,
            IList<PointF> points,
            float divisorX,
            float divisorY,
            bool appendComma)
        {
            writer.WriteLine("  \"" + name + "\": [");

            for (int i = 0; i < points.Count; i++)
            {
                float x = points[i].X / divisorX;
                float y = points[i].Y / divisorY;
                string comma = i < points.Count - 1 ? "," : string.Empty;

                writer.WriteLine(
                    "    { \"x\": " +
                    x.ToString("0.######", CultureInfo.InvariantCulture) +
                    ", \"y\": " +
                    y.ToString("0.######", CultureInfo.InvariantCulture) +
                    " }" +
                    comma);
            }

            writer.WriteLine("  ]" + (appendComma ? "," : string.Empty));
        }

        private static void WriteIntegerArray(
            StreamWriter writer,
            string name,
            IList<int> values,
            bool appendComma)
        {
            writer.Write("  \"" + name + "\": [");

            for (int i = 0; i < values.Count; i++)
            {
                if (i > 0)
                    writer.Write(", ");

                writer.Write(values[i].ToString(CultureInfo.InvariantCulture));
            }

            writer.WriteLine("]" + (appendComma ? "," : string.Empty));
        }

        private static string EscapeJson(string value)
        {
            if (value == null)
                return string.Empty;

            return value
                .Replace("\\", "\\\\")
                .Replace("\"", "\\\"")
                .Replace("\r", "\\r")
                .Replace("\n", "\\n");
        }
    }

    public static class AirDrawComparer
    {
        private const int ComparisonCanvasSize = 256;
        private const int ComparisonPadding = 18;
        private const int BackgroundDifferenceThreshold = 35;
        private const int MinimumComponentPixels = 6;
        private const double FineDistanceSigma = 5d;
        private const double BroadDistanceSigma = 13d;
        private const double StrictCoverageDistance = 5d;
        private const double CalibrationFloor = 0.20d;
        private const double CalibrationCeiling = 0.90d;
        private const double CalibrationPower = 1.15d;
        private const double DiagonalDistance = 1.4142135623730951d;
        private const double InfiniteDistance = 1000000d;
        private static readonly float[] RotationCandidates =
        {
            -8f, -6f, -4f, -2f, 0f, 2f, 4f, 6f, 8f
        };

        public static double Compare(string drawingImagePath, string templateImagePath)
        {
            if (!File.Exists(drawingImagePath))
                throw new FileNotFoundException("Drawing image was not found.", drawingImagePath);

            if (!File.Exists(templateImagePath))
                throw new FileNotFoundException("Template image was not found.", templateImagePath);

            bool[,] drawingMask = LoadNormalizedMask(drawingImagePath);
            bool[,] templateMask = LoadNormalizedMask(templateImagePath);

            int drawingCount = CountPixels(drawingMask);
            int templateCount = CountPixels(templateMask);

            if (drawingCount == 0 || templateCount == 0)
                return 0d;

            double[,] templateDistanceMap = BuildDistanceMap(templateMask);
            double bestRawScore = 0d;

            for (int i = 0; i < RotationCandidates.Length; i++)
            {
                bool[,] rotatedDrawing = RotateMask(
                    drawingMask,
                    RotationCandidates[i]);
                double rawScore = CalculateRawMaskScore(
                    rotatedDrawing,
                    templateMask,
                    templateDistanceMap);

                if (rawScore > bestRawScore)
                    bestRawScore = rawScore;
            }

            return CalibrateScore(bestRawScore);
        }

        private static double CalculateRawMaskScore(
            bool[,] drawingMask,
            bool[,] templateMask,
            double[,] templateDistanceMap)
        {
            int drawingCount = CountPixels(drawingMask);
            int templateCount = CountPixels(templateMask);

            if (drawingCount == 0 || templateCount == 0)
                return 0d;

            double[,] drawingDistanceMap = BuildDistanceMap(drawingMask);
            DistanceStatistics drawingToTemplate = MeasureDistance(
                drawingMask,
                templateDistanceMap);
            DistanceStatistics templateToDrawing = MeasureDistance(
                templateMask,
                drawingDistanceMap);

            double continuousDistanceScore = HarmonicMean(
                drawingToTemplate.SoftCloseness,
                templateToDrawing.SoftCloseness);
            double strictCoverageScore = Math.Min(
                drawingToTemplate.StrictCoverage,
                templateToDrawing.StrictCoverage);

            double worstOutlierDistance = Math.Max(
                drawingToTemplate.Percentile90Distance,
                templateToDrawing.Percentile90Distance);
            double outlierScore = Math.Exp(
                -(worstOutlierDistance * worstOutlierDistance) /
                (2d * BroadDistanceSigma * BroadDistanceSigma));

            double lengthRatio = drawingCount / (double)templateCount;
            double lengthBalanceScore = Math.Exp(
                -1.25d * Math.Abs(Math.Log(lengthRatio)));

            double rawScore =
                continuousDistanceScore * 0.50d +
                strictCoverageScore * 0.22d +
                outlierScore * 0.18d +
                lengthBalanceScore * 0.10d;

            return Clamp01(rawScore);
        }

        private static double CalibrateScore(double rawScore)
        {
            double normalized =
                (rawScore - CalibrationFloor) /
                (CalibrationCeiling - CalibrationFloor);

            normalized = Clamp01(normalized);
            return Math.Pow(normalized, CalibrationPower) * 100d;
        }

        private static double HarmonicMean(double first, double second)
        {
            if (first <= 0d || second <= 0d)
                return 0d;

            return 2d * first * second / (first + second);
        }

        private static double Clamp01(double value)
        {
            return Math.Max(0d, Math.Min(1d, value));
        }

        private static bool[,] RotateMask(
            bool[,] source,
            float angleDegrees)
        {
            if (Math.Abs(angleDegrees) < 0.001f)
                return source;

            int height = source.GetLength(0);
            int width = source.GetLength(1);
            bool[,] result = new bool[height, width];
            double radians = angleDegrees * Math.PI / 180d;
            double cosine = Math.Cos(radians);
            double sine = Math.Sin(radians);
            double centerX = (width - 1) / 2d;
            double centerY = (height - 1) / 2d;

            for (int destinationY = 0;
                destinationY < height;
                destinationY++)
            {
                double dy = destinationY - centerY;

                for (int destinationX = 0;
                    destinationX < width;
                    destinationX++)
                {
                    double dx = destinationX - centerX;
                    int sourceX = (int)Math.Round(
                        cosine * dx + sine * dy + centerX);
                    int sourceY = (int)Math.Round(
                        -sine * dx + cosine * dy + centerY);

                    if (sourceX >= 0 &&
                        sourceX < width &&
                        sourceY >= 0 &&
                        sourceY < height &&
                        source[sourceY, sourceX])
                    {
                        result[destinationY, destinationX] = true;
                    }
                }
            }

            return result;
        }

        private static bool[,] LoadNormalizedMask(string imagePath)
        {
            using (Bitmap source = new Bitmap(imagePath))
            using (Bitmap resized = ResizeWithBackground(source, ComparisonCanvasSize))
            {
                Color background = AverageCornerColor(resized);
                bool[,] rawMask = new bool[ComparisonCanvasSize, ComparisonCanvasSize];

                for (int y = 0; y < ComparisonCanvasSize; y++)
                {
                    for (int x = 0; x < ComparisonCanvasSize; x++)
                    {
                        Color pixel = resized.GetPixel(x, y);
                        int difference =
                            (Math.Abs(pixel.R - background.R) +
                             Math.Abs(pixel.G - background.G) +
                             Math.Abs(pixel.B - background.B)) / 3;

                        bool isStroke = difference >= BackgroundDifferenceThreshold;
                        rawMask[y, x] = isStroke;
                    }
                }

                rawMask = RemoveSmallComponents(
                    rawMask,
                    MinimumComponentPixels);

                int minX;
                int minY;
                int maxX;
                int maxY;

                if (!TryGetBounds(
                    rawMask,
                    out minX,
                    out minY,
                    out maxX,
                    out maxY))
                {
                    return new bool[ComparisonCanvasSize, ComparisonCanvasSize];
                }

                bool[,] normalized = NormalizeMask(
                    rawMask,
                    minX,
                    minY,
                    maxX,
                    maxY);

                return ThinMask(normalized);
            }
        }

        private static Bitmap ResizeWithBackground(Bitmap source, int canvasSize)
        {
            Bitmap result = new Bitmap(canvasSize, canvasSize, PixelFormat.Format24bppRgb);
            Color background = AverageCornerColor(source);

            float scale = Math.Min(
                canvasSize / (float)Math.Max(1, source.Width),
                canvasSize / (float)Math.Max(1, source.Height));

            int width = Math.Max(1, (int)Math.Round(source.Width * scale));
            int height = Math.Max(1, (int)Math.Round(source.Height * scale));
            int x = (canvasSize - width) / 2;
            int y = (canvasSize - height) / 2;

            using (Graphics graphics = Graphics.FromImage(result))
            {
                graphics.Clear(background);
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
                graphics.DrawImage(source, new Rectangle(x, y, width, height));
            }

            return result;
        }

        private static Color AverageCornerColor(Bitmap bitmap)
        {
            Color topLeft = bitmap.GetPixel(0, 0);
            Color topRight = bitmap.GetPixel(Math.Max(0, bitmap.Width - 1), 0);
            Color bottomLeft = bitmap.GetPixel(0, Math.Max(0, bitmap.Height - 1));
            Color bottomRight = bitmap.GetPixel(
                Math.Max(0, bitmap.Width - 1),
                Math.Max(0, bitmap.Height - 1));

            return Color.FromArgb(
                (topLeft.R + topRight.R + bottomLeft.R + bottomRight.R) / 4,
                (topLeft.G + topRight.G + bottomLeft.G + bottomRight.G) / 4,
                (topLeft.B + topRight.B + bottomLeft.B + bottomRight.B) / 4);
        }

        private static bool[,] NormalizeMask(
            bool[,] source,
            int minX,
            int minY,
            int maxX,
            int maxY)
        {
            bool[,] result = new bool[ComparisonCanvasSize, ComparisonCanvasSize];

            int sourceWidth = maxX - minX + 1;
            int sourceHeight = maxY - minY + 1;
            int availableSize = ComparisonCanvasSize - ComparisonPadding * 2;

            float scale = Math.Min(
                availableSize / (float)Math.Max(1, sourceWidth),
                availableSize / (float)Math.Max(1, sourceHeight));

            int destinationWidth = Math.Max(1, (int)Math.Round(sourceWidth * scale));
            int destinationHeight = Math.Max(1, (int)Math.Round(sourceHeight * scale));
            int offsetX = (ComparisonCanvasSize - destinationWidth) / 2;
            int offsetY = (ComparisonCanvasSize - destinationHeight) / 2;

            for (int destinationY = 0; destinationY < destinationHeight; destinationY++)
            {
                int sourceY = minY + Math.Min(
                    sourceHeight - 1,
                    (int)(destinationY / scale));

                for (int destinationX = 0; destinationX < destinationWidth; destinationX++)
                {
                    int sourceX = minX + Math.Min(
                        sourceWidth - 1,
                        (int)(destinationX / scale));

                    if (source[sourceY, sourceX])
                        result[offsetY + destinationY, offsetX + destinationX] = true;
                }
            }

            return result;
        }

        private static int CountPixels(bool[,] mask)
        {
            int count = 0;
            int height = mask.GetLength(0);
            int width = mask.GetLength(1);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (mask[y, x])
                        count++;
                }
            }

            return count;
        }

        private static bool TryGetBounds(
            bool[,] mask,
            out int minX,
            out int minY,
            out int maxX,
            out int maxY)
        {
            int height = mask.GetLength(0);
            int width = mask.GetLength(1);
            minX = width;
            minY = height;
            maxX = -1;
            maxY = -1;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (!mask[y, x])
                        continue;

                    minX = Math.Min(minX, x);
                    minY = Math.Min(minY, y);
                    maxX = Math.Max(maxX, x);
                    maxY = Math.Max(maxY, y);
                }
            }

            return maxX >= minX && maxY >= minY;
        }

        private static bool[,] RemoveSmallComponents(
            bool[,] source,
            int minimumPixels)
        {
            int height = source.GetLength(0);
            int width = source.GetLength(1);
            bool[,] result = (bool[,])source.Clone();
            bool[,] visited = new bool[height, width];
            Queue<Point> pending = new Queue<Point>();
            List<Point> component = new List<Point>();

            for (int startY = 0; startY < height; startY++)
            {
                for (int startX = 0; startX < width; startX++)
                {
                    if (!source[startY, startX] ||
                        visited[startY, startX])
                    {
                        continue;
                    }

                    pending.Clear();
                    component.Clear();
                    pending.Enqueue(new Point(startX, startY));
                    visited[startY, startX] = true;

                    while (pending.Count > 0)
                    {
                        Point current = pending.Dequeue();
                        component.Add(current);

                        for (int offsetY = -1; offsetY <= 1; offsetY++)
                        {
                            for (int offsetX = -1; offsetX <= 1; offsetX++)
                            {
                                if (offsetX == 0 && offsetY == 0)
                                    continue;

                                int nextX = current.X + offsetX;
                                int nextY = current.Y + offsetY;

                                if (nextX < 0 || nextX >= width ||
                                    nextY < 0 || nextY >= height ||
                                    visited[nextY, nextX] ||
                                    !source[nextY, nextX])
                                {
                                    continue;
                                }

                                visited[nextY, nextX] = true;
                                pending.Enqueue(new Point(nextX, nextY));
                            }
                        }
                    }

                    if (component.Count >= minimumPixels)
                        continue;

                    for (int i = 0; i < component.Count; i++)
                    {
                        Point point = component[i];
                        result[point.Y, point.X] = false;
                    }
                }
            }

            return result;
        }

        private static bool[,] ThinMask(bool[,] source)
        {
            bool[,] result = (bool[,])source.Clone();
            List<Point> removalPoints = new List<Point>();
            bool changed;

            do
            {
                bool changedFirst = ApplyThinningPass(
                    result,
                    removalPoints,
                    true);
                bool changedSecond = ApplyThinningPass(
                    result,
                    removalPoints,
                    false);
                changed = changedFirst || changedSecond;
            }
            while (changed);

            return result;
        }

        private static bool ApplyThinningPass(
            bool[,] mask,
            List<Point> removalPoints,
            bool firstPass)
        {
            removalPoints.Clear();
            int height = mask.GetLength(0);
            int width = mask.GetLength(1);

            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    if (!mask[y, x])
                        continue;

                    bool p2 = mask[y - 1, x];
                    bool p3 = mask[y - 1, x + 1];
                    bool p4 = mask[y, x + 1];
                    bool p5 = mask[y + 1, x + 1];
                    bool p6 = mask[y + 1, x];
                    bool p7 = mask[y + 1, x - 1];
                    bool p8 = mask[y, x - 1];
                    bool p9 = mask[y - 1, x - 1];

                    int neighbourCount =
                        (p2 ? 1 : 0) +
                        (p3 ? 1 : 0) +
                        (p4 ? 1 : 0) +
                        (p5 ? 1 : 0) +
                        (p6 ? 1 : 0) +
                        (p7 ? 1 : 0) +
                        (p8 ? 1 : 0) +
                        (p9 ? 1 : 0);

                    if (neighbourCount < 2 || neighbourCount > 6)
                        continue;

                    int transitionCount =
                        (!p2 && p3 ? 1 : 0) +
                        (!p3 && p4 ? 1 : 0) +
                        (!p4 && p5 ? 1 : 0) +
                        (!p5 && p6 ? 1 : 0) +
                        (!p6 && p7 ? 1 : 0) +
                        (!p7 && p8 ? 1 : 0) +
                        (!p8 && p9 ? 1 : 0) +
                        (!p9 && p2 ? 1 : 0);

                    if (transitionCount != 1)
                        continue;

                    bool firstTriplet;
                    bool secondTriplet;

                    if (firstPass)
                    {
                        firstTriplet = p2 && p4 && p6;
                        secondTriplet = p4 && p6 && p8;
                    }
                    else
                    {
                        firstTriplet = p2 && p4 && p8;
                        secondTriplet = p2 && p6 && p8;
                    }

                    if (!firstTriplet && !secondTriplet)
                        removalPoints.Add(new Point(x, y));
                }
            }

            for (int i = 0; i < removalPoints.Count; i++)
            {
                Point point = removalPoints[i];
                mask[point.Y, point.X] = false;
            }

            return removalPoints.Count > 0;
        }

        private static double[,] BuildDistanceMap(bool[,] mask)
        {
            int height = mask.GetLength(0);
            int width = mask.GetLength(1);
            double[,] distances = new double[height, width];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    distances[y, x] =
                        mask[y, x]
                            ? 0d
                            : InfiniteDistance;
                }
            }

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double best = distances[y, x];

                    if (x > 0)
                        best = Math.Min(best, distances[y, x - 1] + 1d);
                    if (y > 0)
                        best = Math.Min(best, distances[y - 1, x] + 1d);
                    if (x > 0 && y > 0)
                    {
                        best = Math.Min(
                            best,
                            distances[y - 1, x - 1] + DiagonalDistance);
                    }
                    if (x + 1 < width && y > 0)
                    {
                        best = Math.Min(
                            best,
                            distances[y - 1, x + 1] + DiagonalDistance);
                    }

                    distances[y, x] = best;
                }
            }

            for (int y = height - 1; y >= 0; y--)
            {
                for (int x = width - 1; x >= 0; x--)
                {
                    double best = distances[y, x];

                    if (x + 1 < width)
                        best = Math.Min(best, distances[y, x + 1] + 1d);
                    if (y + 1 < height)
                        best = Math.Min(best, distances[y + 1, x] + 1d);
                    if (x + 1 < width && y + 1 < height)
                    {
                        best = Math.Min(
                            best,
                            distances[y + 1, x + 1] + DiagonalDistance);
                    }
                    if (x > 0 && y + 1 < height)
                    {
                        best = Math.Min(
                            best,
                            distances[y + 1, x - 1] + DiagonalDistance);
                    }

                    distances[y, x] = best;
                }
            }

            return distances;
        }

        private static DistanceStatistics MeasureDistance(
            bool[,] source,
            double[,] targetDistanceMap)
        {
            List<double> distances = new List<double>();
            double softClosenessSum = 0d;
            int strictMatches = 0;
            int height = source.GetLength(0);
            int width = source.GetLength(1);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (!source[y, x])
                        continue;

                    double distance = targetDistanceMap[y, x];
                    distances.Add(distance);

                    double fineCloseness = Math.Exp(
                        -(distance * distance) /
                        (2d * FineDistanceSigma * FineDistanceSigma));
                    double broadCloseness = Math.Exp(
                        -(distance * distance) /
                        (2d * BroadDistanceSigma * BroadDistanceSigma));

                    softClosenessSum +=
                        fineCloseness * 0.75d +
                        broadCloseness * 0.25d;

                    if (distance <= StrictCoverageDistance)
                        strictMatches++;
                }
            }

            if (distances.Count == 0)
                return new DistanceStatistics();

            distances.Sort();
            int percentileIndex = (int)Math.Ceiling(
                distances.Count * 0.90d) - 1;
            percentileIndex = Math.Max(
                0,
                Math.Min(distances.Count - 1, percentileIndex));

            return new DistanceStatistics
            {
                SoftCloseness =
                    softClosenessSum / distances.Count,
                StrictCoverage =
                    strictMatches / (double)distances.Count,
                Percentile90Distance =
                    distances[percentileIndex]
            };
        }

        private sealed class DistanceStatistics
        {
            public double SoftCloseness { get; set; }
            public double StrictCoverage { get; set; }
            public double Percentile90Distance { get; set; }
        }
    }
}
