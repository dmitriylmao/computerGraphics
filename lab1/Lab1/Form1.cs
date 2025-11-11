using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

public delegate double FunctionDelegate(double x);

namespace Lab1
{
    public partial class Form1 : Form
    {
        private Dictionary<string, FunctionDelegate> availableFunctions;

        public double SinFunction(double x) => Math.Sin(x);
        public double XSquaredFunction(double x) => x * x;
        public double CosFunction(double x) => Math.Cos(x);

        public Form1()
        {
            InitializeComponent();

            availableFunctions = new Dictionary<string, FunctionDelegate>
            {
                {"sin(x)", SinFunction},
                {"x^2", XSquaredFunction},
                {"cos(x)", CosFunction}
            };

            functionComboBox.Items.AddRange(availableFunctions.Keys.ToArray());
            functionComboBox.SelectedIndex = 0;
            //усли форма изменяет размер или выбирается другая функция то вызывается graphPictureBox.Invalidate();
            this.Resize += (s, ev) => graphPictureBox.Invalidate();
            functionComboBox.SelectedIndexChanged += (s, ev) => graphPictureBox.Invalidate();
        }

        private void PlotGraph(Graphics g, int width, int height, FunctionDelegate function)
        {
            double xMin = -10.0;
            double xMax = 10.0;
            double step = (xMax - xMin) / width;

            List<double> yValues = new List<double>();

            for (int i = 0; i < width; i++)
            {
                double x = xMin + i * step;
                double y = function(x);
                yValues.Add(y);
            }

            double yMin = yValues.Min();
            double yMax = yValues.Max();

            if (Math.Abs(yMax - yMin) < 0.0001)
            {
                yMax += 1.0;
                yMin -= 1.0;
            }

            //оси
            Pen axisPen = new Pen(Color.White, 2);

            if (yMin <= 0 && 0 <= yMax)
            {
                float normalizedYZero = (float)((0.0 - yMin) / (yMax - yMin));
                float screenYZero = height - (normalizedYZero * height);
                g.DrawLine(axisPen, 0, screenYZero, width, screenYZero);
            }

            if (xMin <= 0 && 0 <= xMax)
            {
                float normalizedXZero = (float)((0.0 - xMin) / (xMax - xMin));
                float screenXZero = normalizedXZero * width;
                g.DrawLine(axisPen, screenXZero, 0, screenXZero, height);
            }

            //график
            Pen graphPen = new Pen(Color.Red, 4);
            PointF previousPoint = PointF.Empty; //предыдущая точка

            for (int i = 0; i < width; i++)
            {
                double yValue = yValues[i];

                float screenX = i;
                float normalizedY = (float)((yValue - yMin) / (yMax - yMin));
                float screenY = height - (normalizedY * height);

                PointF currentPoint = new PointF(screenX, screenY);

                if (i > 0)
                {
                    g.DrawLine(graphPen, previousPoint, currentPoint);
                }
                previousPoint = currentPoint;
            }
        }

        private void graphPictureBox_Paint(object sender, PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            int width = graphPictureBox.Width;
            int height = graphPictureBox.Height;

            string selectedKey = functionComboBox.SelectedItem.ToString();
            FunctionDelegate currentFunction = availableFunctions[selectedKey];

            PlotGraph(g, width, height, currentFunction);
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }
    }
}