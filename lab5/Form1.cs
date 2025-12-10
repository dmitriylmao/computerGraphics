using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.Rebar;

namespace Lab4_Bezier
{
    public partial class Form1 : Form
    {
        private readonly List<PointF> _controlPoints = new List<PointF>();
        private int _draggingIndex = -1;       // индекс точки, которую двигаем (-1 = ничего)
        private const float PointRadius = 5f;  // радиус отрисовки точки
        private const float HitRadius = 8f;    // радиус попадания мышью

        public Form1()
        {
            InitializeComponent();

            this.DoubleBuffered = true;
        }

        private void canvasPictureBox_Paint(object sender, PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;

            g.Clear(Color.White);

            //рисуем ломаную по опорным точкам
            if (_controlPoints.Count > 1)
            {
                using (var pen = new Pen(Color.LightGray, 1f))
                {
                    for (int i = 0; i < _controlPoints.Count - 1; i++)
                    {
                        g.DrawLine(pen, _controlPoints[i], _controlPoints[i + 1]);
                    }
                }
            }

            // рисуем сами опорные точки
            using (var brush = new SolidBrush(Color.Red))
            using (var penPoint = new Pen(Color.Black, 1f))
            {
                foreach (var p in _controlPoints)
                {
                    RectangleF rect = new RectangleF(
                        p.X - PointRadius,
                        p.Y - PointRadius,
                        PointRadius * 2,
                        PointRadius * 2);

                    g.FillEllipse(brush, rect);
                    g.DrawEllipse(penPoint, rect);
                }
            }

            // рисуем составную кубическую кривую Безье
            if (_controlPoints.Count >= 4)
            {
                using (var curvePen = new Pen(Color.Blue, 2f))
                {
                    int segments = (_controlPoints.Count - 1) / 3; // каждый новый сегмент добавляет 3 точки

                    for (int s = 0; s < segments; s++)
                    {
                        int i0 = s * 3;
                        PointF p0 = _controlPoints[i0];
                        PointF p1 = _controlPoints[i0 + 1];
                        PointF p2 = _controlPoints[i0 + 2];
                        PointF p3 = _controlPoints[i0 + 3];

                        DrawBezierSegment(g, curvePen, p0, p1, p2, p3);
                    }
                }
            }
        }

        private void DrawBezierSegment(Graphics g, Pen pen, PointF p0, PointF p1, PointF p2, PointF p3)
        {
            const int steps = 50;
            float dt = 1f / steps;

            PointF prev = BezierPoint(p0, p1, p2, p3, 0f);

            for (int i = 1; i <= steps; i++)
            {
                float t = dt * i;
                PointF curr = BezierPoint(p0, p1, p2, p3, t);
                g.DrawLine(pen, prev, curr);
                prev = curr;
            }
        }

        private PointF BezierPoint(PointF p0, PointF p1, PointF p2, PointF p3, float t) //формула кубической кривой безье
        {
            float u = 1 - t;
            float tt = t * t;
            float uu = u * u;
            float uuu = uu * u;
            float ttt = tt * t;

            float x =
                uuu * p0.X +
                3 * uu * t * p1.X +
                3 * u * tt * p2.X +
                ttt * p3.X;

            float y =
                uuu * p0.Y +
                3 * uu * t * p1.Y +
                3 * u * tt * p2.Y +
                ttt * p3.Y;

            return new PointF(x, y);
        }


        private void canvasPictureBox_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left)
                return;

            if (rbMove.Checked)
            {
                //ищем ближайшую точку
                int idx = FindPointIndex(e.Location, HitRadius);
                if (idx != -1)
                {
                    _draggingIndex = idx;
                }
            }
            else if (rbAdd.Checked)
            {
                //добавляем точку
                _controlPoints.Add(e.Location);
                canvasPictureBox.Invalidate();
            }
            else if (rbDelete.Checked)
            {
                //удаляем ближайшую к клику точку
                int idx = FindPointIndex(e.Location, HitRadius);
                if (idx != -1)
                {
                    _controlPoints.RemoveAt(idx);
                    canvasPictureBox.Invalidate();
                }
            }
        }

        private void canvasPictureBox_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left)
                return;

            if (_draggingIndex >= 0 && rbMove.Checked)
            {
                // двигаем выбранную точку за мышью
                _controlPoints[_draggingIndex] = e.Location;
                canvasPictureBox.Invalidate();
            }
        }

        private void canvasPictureBox_MouseUp(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                _draggingIndex = -1;
            }
        }

        private int FindPointIndex(Point location, float radius)
        {
            float r2 = radius * radius;

            for (int i = 0; i < _controlPoints.Count; i++)
            {
                float dx = _controlPoints[i].X - location.X;
                float dy = _controlPoints[i].Y - location.Y;
                float dist2 = dx * dx + dy * dy;

                if (dist2 <= r2)
                    return i;
            }

            return -1;
        }

        private void btnClear_Click(object sender, EventArgs e)
        {
            _controlPoints.Clear();
            _draggingIndex = -1;
            canvasPictureBox.Invalidate();
        }
    }
}
