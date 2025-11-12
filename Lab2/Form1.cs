using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Lab1
{
    public partial class Form1 : Form
    {
        private Bitmap originalImage;
        private Bitmap processedImage;

        private int hueShift = 0;
        private int satShift = 0;
        private int valShift = 0;

        public Form1()
        {
            InitializeComponent();

            hueTrackBar.Scroll += HsvTrackBar_Scroll;
            satTrackBar.Scroll += HsvTrackBar_Scroll;
            valTrackBar.Scroll += HsvTrackBar_Scroll;

            UpdateLabels();
            saveButton.Enabled = false;
        }

        private void UpdateLabels()
        {
            hueLabel.Text = $"H (Оттенок): {hueTrackBar.Value}°";
            satLabel.Text = $"S (Насыщенность): {satTrackBar.Value}%";
            valLabel.Text = $"V (Яркость): {valTrackBar.Value}%";
        }

        private void HsvTrackBar_Scroll(object sender, EventArgs e)
        {
            hueShift = hueTrackBar.Value;
            satShift = satTrackBar.Value;
            valShift = valTrackBar.Value;

            UpdateLabels();

            if (originalImage != null)
            {
                ProcessImageFast();
            }
        }

        private void loadButton_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog ofd = new OpenFileDialog())
            {
                ofd.Filter = "Image Files|*.bmp;*.jpg;*.jpeg;*.png;*.gif";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    originalImage = new Bitmap(ofd.FileName);
                    sourcePictureBox.Image = originalImage;

                    ProcessImageFast();

                    saveButton.Enabled = true;
                }
            }
        }

        private void saveButton_Click(object sender, EventArgs e)
        {
            if (processedImage == null) return;

            using (SaveFileDialog sfd = new SaveFileDialog())
            {
                sfd.Filter = "PNG Image|*.png|JPEG Image|*.jpg";
                if (sfd.ShowDialog() == DialogResult.OK)
                {
                    processedImage.Save(sfd.FileName);
                }
            }
        }

        private void ProcessImageFast()
        {
            if (originalImage == null) return;

            int width = originalImage.Width;
            int height = originalImage.Height;

            processedImage = (Bitmap)originalImage.Clone();

            Rectangle rect = new Rectangle(0, 0, width, height);
            //получаем доступ к пикселям
            BitmapData bmpData = processedImage.LockBits(rect, ImageLockMode.ReadWrite, processedImage.PixelFormat);

            IntPtr ptr = bmpData.Scan0;

            int bytes = Math.Abs(bmpData.Stride) * height;
            byte[] rgbValues = new byte[bytes];

            Marshal.Copy(ptr, rgbValues, 0, bytes);

            int pixelSize = Image.GetPixelFormatSize(processedImage.PixelFormat) / 8;

            for (int i = 0; i < bytes; i += pixelSize)
            {
                int b = rgbValues[i];
                int g = rgbValues[i + 1];
                int r = rgbValues[i + 2];

                RgbToHsv(r, g, b, out double h, out double s, out double v);

                double hPrime = h + hueShift;
                double sPrime = s + (double)satShift / 100.0;
                double vPrime = v + (double)valShift / 100.0;

                if (hPrime > 360) hPrime -= 360;
                if (hPrime < 0) hPrime += 360;
                sPrime = Math.Max(0.0, Math.Min(1.0, sPrime));
                vPrime = Math.Max(0.0, Math.Min(1.0, vPrime));

                Color newColor = HsvToRgb(hPrime, sPrime, vPrime);

                rgbValues[i] = newColor.B;
                rgbValues[i + 1] = newColor.G;
                rgbValues[i + 2] = newColor.R;
            }

            Marshal.Copy(rgbValues, 0, ptr, bytes);

            processedImage.UnlockBits(bmpData);

            resultPictureBox.Image = processedImage;
        }

        private void RgbToHsv(int r, int g, int b, out double h, out double s, out double v)
        {
            double rNorm = r / 255.0;
            double gNorm = g / 255.0;
            double bNorm = b / 255.0;

            double max = Math.Max(rNorm, Math.Max(gNorm, bNorm));
            double min = Math.Min(rNorm, Math.Min(gNorm, bNorm));
            double delta = max - min;

            v = max;

            if (max == 0.0)
            {
                s = 0;
                h = 0;
                return;
            }

            s = delta / max;

            if (delta == 0.0)
            {
                h = 0;
                return;
            }

            if (max == rNorm)
            {
                h = 60.0 * ((gNorm - bNorm) / delta);
            }
            else if (max == gNorm)
            {
                h = 60.0 * ((bNorm - rNorm) / delta + 2.0);
            }
            else
            {
                h = 60.0 * ((rNorm - gNorm) / delta + 4.0);
            }

            if (h < 0.0)
            {
                h += 360.0;
            }
        }

        private Color HsvToRgb(double h, double s, double v)
        {
            double r = 0, g = 0, b = 0;

            if (s == 0)
            {
                r = g = b = v;
            }
            else
            {
                double sector = h / 60.0;
                int i = (int)Math.Floor(sector);
                double f = sector - i;
                double p = v * (1.0 - s);
                double q = v * (1.0 - s * f);
                double t = v * (1.0 - s * (1.0 - f));

                switch (i)
                {
                    case 0: r = v; g = t; b = p; break;
                    case 1: r = q; g = v; b = p; break;
                    case 2: r = p; g = v; b = t; break;
                    case 3: r = p; g = q; b = v; break;
                    case 4: r = t; g = p; b = v; break;
                    default: r = v; g = p; b = q; break;
                }
            }

            return Color.FromArgb(
                (int)Math.Round(Math.Min(255.0, r * 255.0)),
                (int)Math.Round(Math.Min(255.0, g * 255.0)),
                (int)Math.Round(Math.Min(255.0, b * 255.0))
            );
        }
    }
}