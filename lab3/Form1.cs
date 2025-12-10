using System;
using System.Drawing;
using System.Windows.Forms;

namespace Lab3_RasterLines
{
    public partial class Form1 : Form
    {
        private Bitmap _bitmap;
        private Point? _firstPoint;          // первая точка отрезка
        private readonly Color _lineColor = Color.Black;

        public Form1()
        {
            InitializeComponent();
            InitCanvas();

            // Пересоздаём холст при изменении размеров окна
            this.Resize += Form1_Resize;
        }

        private void InitCanvas()
        {
            if (canvasPictureBox.Width <= 0 || canvasPictureBox.Height <= 0)
                return;

            _bitmap?.Dispose();
            _bitmap = new Bitmap(canvasPictureBox.Width, canvasPictureBox.Height);
            using (var g = Graphics.FromImage(_bitmap))
            {
                g.Clear(Color.White);
            }
            canvasPictureBox.Image = _bitmap;
        }
        private void Form1_Resize(object sender, EventArgs e)
        {
            InitCanvas();
        }


        // Обработка клика по полю
        private void canvasPictureBox_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left || _bitmap == null)
                return;

            if (_firstPoint == null)
            {
                _firstPoint = e.Location; //запоминаем позицию клика e.Location как начало отрезка
            }
            else
            {
                var p0 = _firstPoint.Value;
                var p1 = e.Location;

                DrawLine(p0.X, p0.Y, p1.X, p1.Y);

                _firstPoint = null;
                canvasPictureBox.Invalidate();
            }
        }

        //Очистка поля
        private void btnClear_Click(object sender, EventArgs e)
        {
            if (_bitmap == null)
                return;

            using (var g = Graphics.FromImage(_bitmap))
            {
                g.Clear(Color.White);
            }
            _firstPoint = null;
            canvasPictureBox.Invalidate();
        }

        private void DrawLine(int x0, int y0, int x1, int y1)
        {
            if (rbBresenham.Checked)
                DrawLineBresenham(x0, y0, x1, y1);
            else
                DrawLineWu(x0, y0, x1, y1);
        }

        #region Брезенхем (целочисленный)

        // Кладём пиксель с проверкой границ
        private void PutPixel(int x, int y, Color color)
        {
            if (_bitmap == null)
                return;

            if (x < 0 || x >= _bitmap.Width || y < 0 || y >= _bitmap.Height)
                return;

            _bitmap.SetPixel(x, y, color);
        }

        private static void Swap<T>(ref T a, ref T b)
        {
            T tmp = a;
            a = b;
            b = tmp;
        }

        // Целочисленный Брезенхем со сменой осей для крутых наклонов
        private void DrawLineBresenham(int x0, int y0, int x1, int y1)
        {
            bool steep = Math.Abs(y1 - y0) > Math.Abs(x1 - x0); //линия «крутая», если вертикальный разброс |dy| больше горизонтального |dx|

            // если линия крутая — меняем местами x и y
            if (steep)
            {
                Swap(ref x0, ref y0);
                Swap(ref x1, ref y1);
            }

            // рисуем слева направо
            if (x0 > x1)
            {
                Swap(ref x0, ref x1);
                Swap(ref y0, ref y1);
            }

            int dx = x1 - x0;
            int dy = Math.Abs(y1 - y0);
            int error = dx / 2;                     
            int yStep = (y0 < y1) ? 1 : -1;
            int y = y0;

            for (int x = x0; x <= x1; x++)
            {
                if (steep)
                    PutPixel(y, x, _lineColor);
                else
                    PutPixel(x, y, _lineColor);

                error -= dy;
                if (error < 0)
                {
                    y += yStep;
                    error += dx;
                }
            }
        }

        #endregion

        #region Алгоритм Ву (сглаженная линия)

        // Рисуем пиксель с "интенсивностью" 0..1 (0 — нет линии, 1 — чёрный)
        private void PlotWu(int x, int y, double intensity)
        {
            if (_bitmap == null)
                return;

            if (x < 0 || x >= _bitmap.Width || y < 0 || y >= _bitmap.Height)
                return;

            if (intensity < 0.0) intensity = 0.0;
            if (intensity > 1.0) intensity = 1.0;

            // Фон белый, линия чёрная: чем больше intensity, тем темнее пиксель
            int value = 255 - (int)(intensity * 255.0);
            if (value < 0) value = 0;
            if (value > 255) value = 255;

            Color c = Color.FromArgb(value, value, value);
            _bitmap.SetPixel(x, y, c);
        }

        private static double FPart(double x) => x - Math.Floor(x);
        private static double RFPart(double x) => 1.0 - FPart(x);

        private void DrawLineWu(int x0, int y0, int x1, int y1)
        {
            bool steep = Math.Abs(y1 - y0) > Math.Abs(x1 - x0);

            if (steep)
            {
                Swap(ref x0, ref y0);
                Swap(ref x1, ref y1);
            }

            if (x0 > x1)
            {
                Swap(ref x0, ref x1);
                Swap(ref y0, ref y1);
            }

            double dx = x1 - x0;
            double dy = y1 - y0;
            double gradient = dx == 0.0 ? 0.0 : dy / dx;

            // Первый конец
            double xEnd = (double)x0; // был Math.Round(x0)
            double yEnd = y0 + gradient * (xEnd - x0);
            double xGap = RFPart(x0 + 0.5);
            int xPxl1 = (int)xEnd;
            int yPxl1 = (int)Math.Floor(yEnd);

            if (steep)
            {
                PlotWu(yPxl1, xPxl1, RFPart(yEnd) * xGap);
                PlotWu(yPxl1 + 1, xPxl1, FPart(yEnd) * xGap);
            }
            else
            {
                PlotWu(xPxl1, yPxl1, RFPart(yEnd) * xGap);
                PlotWu(xPxl1, yPxl1 + 1, FPart(yEnd) * xGap);
            }

            double intery = yEnd + gradient;

            // Второй конец
            xEnd = (double)x1; // был Math.Round(x1)
            yEnd = y1 + gradient * (xEnd - x1);
            xGap = FPart(x1 + 0.5);
            int xPxl2 = (int)xEnd;
            int yPxl2 = (int)Math.Floor(yEnd);

            if (steep)
            {
                PlotWu(yPxl2, xPxl2, RFPart(yEnd) * xGap);
                PlotWu(yPxl2 + 1, xPxl2, FPart(yEnd) * xGap);
            }
            else
            {
                PlotWu(xPxl2, yPxl2, RFPart(yEnd) * xGap);
                PlotWu(xPxl2, yPxl2 + 1, FPart(yEnd) * xGap);
            }

            // Основной цикл
            if (steep)
            {
                for (int x = xPxl1 + 1; x <= xPxl2 - 1; x++)
                {
                    int y = (int)Math.Floor(intery);
                    PlotWu(y, x, RFPart(intery));
                    PlotWu(y + 1, x, FPart(intery));
                    intery += gradient;
                }
            }
            else
            {
                for (int x = xPxl1 + 1; x <= xPxl2 - 1; x++)
                {
                    int y = (int)Math.Floor(intery);
                    PlotWu(x, y, RFPart(intery));
                    PlotWu(x, y + 1, FPart(intery));
                    intery += gradient;
                }
            }
        }

        #endregion
    }
}
