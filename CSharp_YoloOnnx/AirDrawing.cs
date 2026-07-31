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
        private const int MatchTolerance = 6;
        private const int BackgroundDifferenceThreshold = 35;

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

            int matchedDrawing = CountMatchedPixels(drawingMask, templateMask, MatchTolerance);
            int matchedTemplate = CountMatchedPixels(templateMask, drawingMask, MatchTolerance);

            double precision = matchedDrawing / (double)drawingCount;
            double recall = matchedTemplate / (double)templateCount;

            if (precision + recall <= 0d)
                return 0d;

            double score = 2d * precision * recall / (precision + recall) * 100d;
            return Math.Max(0d, Math.Min(100d, score));
        }

        private static bool[,] LoadNormalizedMask(string imagePath)
        {
            using (Bitmap source = new Bitmap(imagePath))
            using (Bitmap resized = ResizeWithBackground(source, ComparisonCanvasSize))
            {
                Color background = AverageCornerColor(resized);
                bool[,] rawMask = new bool[ComparisonCanvasSize, ComparisonCanvasSize];

                int minX = ComparisonCanvasSize;
                int minY = ComparisonCanvasSize;
                int maxX = -1;
                int maxY = -1;

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

                        if (isStroke)
                        {
                            minX = Math.Min(minX, x);
                            minY = Math.Min(minY, y);
                            maxX = Math.Max(maxX, x);
                            maxY = Math.Max(maxY, y);
                        }
                    }
                }

                if (maxX < minX || maxY < minY)
                    return new bool[ComparisonCanvasSize, ComparisonCanvasSize];

                return NormalizeMask(rawMask, minX, minY, maxX, maxY);
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

        private static int CountMatchedPixels(bool[,] source, bool[,] target, int tolerance)
        {
            int matched = 0;
            int height = source.GetLength(0);
            int width = source.GetLength(1);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (source[y, x] && HasNearbyPixel(target, x, y, tolerance))
                        matched++;
                }
            }

            return matched;
        }

        private static bool HasNearbyPixel(bool[,] mask, int centerX, int centerY, int tolerance)
        {
            int height = mask.GetLength(0);
            int width = mask.GetLength(1);
            int minX = Math.Max(0, centerX - tolerance);
            int maxX = Math.Min(width - 1, centerX + tolerance);
            int minY = Math.Max(0, centerY - tolerance);
            int maxY = Math.Min(height - 1, centerY + tolerance);
            int toleranceSquared = tolerance * tolerance;

            for (int y = minY; y <= maxY; y++)
            {
                for (int x = minX; x <= maxX; x++)
                {
                    int dx = x - centerX;
                    int dy = y - centerY;

                    if (dx * dx + dy * dy <= toleranceSquared && mask[y, x])
                        return true;
                }
            }

            return false;
        }
    }
}
