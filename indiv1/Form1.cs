using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

namespace Lab5_Graham
{
    public partial class Form1 : Form
    {
        private readonly List<PointF> _points = new List<PointF>();
        private readonly List<PointF> _hull = new List<PointF>();
        private readonly Random _rnd = new Random();


        public Form1()
        {
            InitializeComponent();
            this.DoubleBuffered = true;
        }

        private void canvasPictureBox_Paint(object sender, PaintEventArgs e)
        {
            var g = e.Graphics;
            g.Clear(Color.White);
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

            const float r = 4f;
            using (var ptBrush = new SolidBrush(Color.Black))
            using (var ptPen = new Pen(Color.Black, 1f))
            {
                foreach (var p in _points)
                {
                    g.FillEllipse(ptBrush, p.X - r, p.Y - r, 2 * r, 2 * r);
                    g.DrawEllipse(ptPen, p.X - r, p.Y - r, 2 * r, 2 * r);
                }
            }

            if (_hull.Count >= 2)
            {
                using (var hullPen = new Pen(Color.Red, 2f))
                {
                    if (_hull.Count >= 3)
                        g.DrawPolygon(hullPen, _hull.ToArray());
                    else
                        g.DrawLines(hullPen, _hull.ToArray());
                }
            }
        }

        // ===== МЫШЬ: ДОБАВЛЕНИЕ ТОЧЕК =======================================

        private void canvasPictureBox_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left)
                return;

            // добавляем точку кликом
            _points.Add(e.Location);
            _hull.Clear();              // при изменении набора точек оболочка сбрасывается
            canvasPictureBox.Invalidate();
            UpdateStatus();
        }

        // ===== КНОПКИ ========================================================

        private void btnClear_Click(object sender, EventArgs e)
        {
            _points.Clear();
            _hull.Clear();
            canvasPictureBox.Invalidate();
            UpdateStatus();
        }

        private void btnRandom_Click(object sender, EventArgs e)
        {
            _points.Clear();
            _hull.Clear();

            int w = canvasPictureBox.Width - 20;
            int h = canvasPictureBox.Height - 20;

            int n = 20; // кол-во случайных точек
            for (int i = 0; i < n; i++)
            {
                float x = 10 + (float)_rnd.NextDouble() * w;
                float y = 10 + (float)_rnd.NextDouble() * h;
                _points.Add(new PointF(x, y));
            }

            canvasPictureBox.Invalidate();
            UpdateStatus();
        }

        private void btnBuildHull_Click(object sender, EventArgs e)
        {
            _hull.Clear();

            if (_points.Count < 3)
            {
                MessageBox.Show("Для выпуклой оболочки нужно минимум 3 точки.",
                    "Грэхем", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            var hull = BuildConvexHullGraham(_points);
            _hull.AddRange(hull);

            canvasPictureBox.Invalidate();
            UpdateStatus();
        }

        private void UpdateStatus()
        {
            lblStatus.Text = $"Точек: {_points.Count}, в оболочке: {_hull.Count}";
        }

        // ===== АЛГОРИТМ ГРЭХЭМА ==============================================

        private static List<PointF> BuildConvexHullGraham(List<PointF> input)
        {
            // копия списка
            var points = input.ToList();

            // 1. Находим опорную точку: минимальный Y, при равенстве — минимальный X
            int pivotIndex = 0;
            for (int i = 1; i < points.Count; i++)
            {
                if (points[i].Y < points[pivotIndex].Y ||
                    (Math.Abs(points[i].Y - points[pivotIndex].Y) < 1e-6 &&
                     points[i].X < points[pivotIndex].X))
                {
                    pivotIndex = i;
                }
            }

            PointF pivot = points[pivotIndex];
            points.RemoveAt(pivotIndex);

            // 2. Сортируем остальные по полярному углу относительно pivot
            points.Sort((a, b) =>
            {
                double angleA = Math.Atan2(a.Y - pivot.Y, a.X - pivot.X);
                double angleB = Math.Atan2(b.Y - pivot.Y, b.X - pivot.X);

                int cmp = angleA.CompareTo(angleB);
                if (cmp != 0)
                    return cmp;

                // если угол одинаковый — дальше по расстоянию от pivot
                double distA = Dist2(pivot, a);
                double distB = Dist2(pivot, b);
                return distA.CompareTo(distB);
            });

            // 3. Проход по отсортированным точкам со стеком
            var hull = new List<PointF> { pivot };

            foreach (var pt in points)
            {
                while (hull.Count >= 2 &&
                       Cross(hull[hull.Count - 2], hull[hull.Count - 1], pt) <= 0)
                {
                    // удаляем «правый» поворот или коллинеарность
                    hull.RemoveAt(hull.Count - 1);
                }
                hull.Add(pt);
            }

            return hull;
        }

        // Векторное произведение (AB x AC). >0 — левый поворот, <0 — правый
        private static double Cross(PointF a, PointF b, PointF c)
        {
            double abx = b.X - a.X;
            double aby = b.Y - a.Y;
            double acx = c.X - a.X;
            double acy = c.Y - a.Y;
            return abx * acy - aby * acx;
        }

        private static double Dist2(PointF a, PointF b)
        {
            double dx = a.X - b.X;
            double dy = a.Y - b.Y;
            return dx * dx + dy * dy;
        }
    }
}
