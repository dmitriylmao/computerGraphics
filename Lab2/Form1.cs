using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace Lab1
{
    public partial class Form1 : Form
    {
        private Bitmap originalImage;
        private Bitmap processedImage;

        private int hueShift;   
        private int satShift; 
        private int valShift; 

        public Form1()
        {
            InitializeComponent();
            InitControls();
        }

        private void InitControls()
        {
            hueTrackBar.Minimum = -180;
            hueTrackBar.Maximum = 180;

            satTrackBar.Minimum = -100;
            satTrackBar.Maximum = 100;

            valTrackBar.Minimum = -100;
            valTrackBar.Maximum = 100;

            hueTrackBar.Value = 0;
            satTrackBar.Value = 0;
            valTrackBar.Value = 0;

            hueTrackBar.Scroll += HsvTrackBar_Scroll;
            satTrackBar.Scroll += HsvTrackBar_Scroll;
            valTrackBar.Scroll += HsvTrackBar_Scroll;

            //только при отпускании мыши
            hueTrackBar.MouseUp += TrackBar_MouseUp;
            satTrackBar.MouseUp += TrackBar_MouseUp;
            valTrackBar.MouseUp += TrackBar_MouseUp;

            UpdateShiftsFromTrackBars();
            UpdateLabels();
        }

        private void loadButton_Click(object sender, EventArgs e)
        {
            using (var ofd = new OpenFileDialog())
            {
                ofd.Filter = "Image files|*.bmp;*.jpg;*.jpeg;*.png|All files|*.*";

                if (ofd.ShowDialog() != DialogResult.OK)
                    return;

                // чистим старые картинки
                originalImage?.Dispose();
                processedImage?.Dispose();

                originalImage = (Bitmap)Image.FromFile(ofd.FileName);
                processedImage = null;

                sourcePictureBox.Image = originalImage;
                resultPictureBox.Image = null;

                ResetShifts();
                ProcessImageFast();
            }
        }

        private void saveButton_Click(object sender, EventArgs e)
        {
            if (processedImage == null)
            {
                MessageBox.Show("Нет обработанного изображения для сохранения.",
                                "Сохранение",
                                MessageBoxButtons.OK,
                                MessageBoxIcon.Information);
                return;
            }

            using (var sfd = new SaveFileDialog())
            {
                sfd.Filter = "PNG image|*.png|JPEG image|*.jpg;*.jpeg|Bitmap image|*.bmp";
                sfd.DefaultExt = "png";

                if (sfd.ShowDialog() != DialogResult.OK)
                    return;

                var ext = Path.GetExtension(sfd.FileName)?.ToLowerInvariant();
                ImageFormat format = ImageFormat.Png;

                if (ext == ".jpg" || ext == ".jpeg")
                    format = ImageFormat.Jpeg;
                else if (ext == ".bmp")
                    format = ImageFormat.Bmp;

                processedImage.Save(sfd.FileName, format);
            }
        }

        // Сброс ползунков в ноль
        private void ResetShifts()
        {
            hueTrackBar.Value = 0;
            satTrackBar.Value = 0;
            valTrackBar.Value = 0;

            UpdateShiftsFromTrackBars();
            UpdateLabels();
        }

        private void HsvTrackBar_Scroll(object sender, EventArgs e)
        {
            UpdateShiftsFromTrackBars();
            UpdateLabels();
        }

        // Отпустили мышь на любом ползунке — считаем картинку
        private void TrackBar_MouseUp(object sender, MouseEventArgs e)
        {
            if (originalImage != null)
            {
                ProcessImageFast();
            }
        }

        private void UpdateShiftsFromTrackBars()
        {
            hueShift = hueTrackBar.Value;
            satShift = satTrackBar.Value;
            valShift = valTrackBar.Value;
        }

        private void UpdateLabels()
        {
            hueLabel.Text = $"Hue: {hueShift}°";
            satLabel.Text = $"Sat: {satShift}%";
            valLabel.Text = $"Val: {valShift}%";
        }


        private void ProcessImageFast()
        {
            if (originalImage == null)
                return;

            int width = originalImage.Width;
            int height = originalImage.Height;

            processedImage?.Dispose();
            processedImage = (Bitmap)originalImage.Clone();

            var rect = new Rectangle(0, 0, width, height);
            BitmapData bmpData = processedImage.LockBits(
                rect,
                ImageLockMode.ReadWrite,
                processedImage.PixelFormat);


            int stride = bmpData.Stride;
            int bytes = Math.Abs(stride) * height;
            byte[] rgbValues = new byte[bytes];

            Marshal.Copy(bmpData.Scan0, rgbValues, 0, bytes);

            int pixelSize = Image.GetPixelFormatSize(processedImage.PixelFormat) / 8;
            if (pixelSize < 3)
            {
                processedImage.UnlockBits(bmpData);
                return;
            }

            double satAdd = satShift / 100.0;
            double valAdd = valShift / 100.0;

            for (int y = 0; y < height; y++)
            {
                int rowStart = y * stride;

                for (int x = 0; x < width; x++)
                {
                    int i = rowStart + x * pixelSize;

                    if (i + 2 >= rgbValues.Length)
                        continue;

                    int b = rgbValues[i];
                    int g = rgbValues[i + 1];
                    int r = rgbValues[i + 2];

                    RgbToHsv(r, g, b, out double h, out double s, out double v);

                    double hPrime = h + hueShift;
                    double sPrime = s + satAdd;
                    double vPrime = v + valAdd;

                    hPrime %= 360.0;
                    if (hPrime < 0)
                        hPrime += 360.0;

                    if (sPrime < 0.0) sPrime = 0.0;
                    if (sPrime > 1.0) sPrime = 1.0;
                    if (vPrime < 0.0) vPrime = 0.0;
                    if (vPrime > 1.0) vPrime = 1.0;

                    Color newColor = HsvToRgb(hPrime, sPrime, vPrime);

                    rgbValues[i] = newColor.B;
                    rgbValues[i + 1] = newColor.G;
                    rgbValues[i + 2] = newColor.R;
                }
            }

            Marshal.Copy(rgbValues, 0, bmpData.Scan0, bytes); //Возвращаем байты в картинку
            processedImage.UnlockBits(bmpData);

            resultPictureBox.Image = processedImage;
        }

        //(0..255) ->(H:0..360, S,V:0..1)
        private static void RgbToHsv(int r, int g, int b, out double h, out double s, out double v)
        {
            double rf = r / 255.0; //нормирование в 0 1
            double gf = g / 255.0;
            double bf = b / 255.0;

            double max = Math.Max(rf, Math.Max(gf, bf));
            double min = Math.Min(rf, Math.Min(gf, bf));
            double delta = max - min;

            v = max;

            if (delta < 1e-6)
            {
                h = 0.0;
                s = 0.0; //серый цвец
                return;
            }

            s = delta / max;

            if (Math.Abs(max - rf) < 1e-6)
            {
                h = 60.0 * ((gf - bf) / delta);
            }
            else if (Math.Abs(max - gf) < 1e-6)                 //узнаем какой цвет преобладает , чтобы попасть в нужный сектор круга 
            {
                h = 60.0 * (((bf - rf) / delta) + 2.0);
            }
            else
            {
                h = 60.0 * (((rf - gf) / delta) + 4.0);
            }

            if (h < 0.0)
                h += 360.0;
        }

        // HSV -> RGB
        private static Color HsvToRgb(double h, double s, double v)
        {
            if (s <= 0.0) //если сатурация примерно 0 , то это оттенок серого
            {
                int value = ClampToByte(v * 255.0);
                return Color.FromArgb(value, value, value);
            }

            h = h % 360.0;
            if (h < 0.0)
                h += 360.0;

            double c = v * s;
            double x = c * (1.0 - Math.Abs((h / 60.0) % 2.0 - 1.0));
            double m = v - c;

            double rf, gf, bf;

            if (h < 60.0)
            {
                rf = c; gf = x; bf = 0.0;
            }
            else if (h < 120.0)
            {
                rf = x; gf = c; bf = 0.0;
            }
            else if (h < 180.0)
            {
                rf = 0.0; gf = c; bf = x;
            }
            else if (h < 240.0)
            {
                rf = 0.0; gf = x; bf = c;
            }
            else if (h < 300.0)
            {
                rf = x; gf = 0.0; bf = c;
            }
            else
            {
                rf = c; gf = 0.0; bf = x;
            }

            int r = ClampToByte((rf + m) * 255.0);
            int g = ClampToByte((gf + m) * 255.0);
            int b = ClampToByte((bf + m) * 255.0);

            return Color.FromArgb(r, g, b);
        }

        private static int ClampToByte(double value) //загоняем дабл в 0...255 и округляем
        {
            if (value < 0.0) return 0;
            if (value > 255.0) return 255;
            return (int)Math.Round(value);
        }
    }
}
