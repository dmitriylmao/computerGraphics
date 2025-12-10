using System.Windows.Forms;
using System.Drawing;

namespace Lab3_RasterLines
{
    partial class Form1
    {
        /// <summary>
        /// Обязательная переменная конструктора.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        private PictureBox canvasPictureBox;
        private Panel panelRight;
        private RadioButton rbBresenham;
        private RadioButton rbWu;
        private Button btnClear;
        private Label infoLabel;

        /// <summary>
        /// Освобождение ресурсов.
        /// </summary>
        /// <param name="disposing"></param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Код конструктора формы

        private void InitializeComponent()
        {
            this.canvasPictureBox = new System.Windows.Forms.PictureBox();
            this.panelRight = new System.Windows.Forms.Panel();
            this.rbBresenham = new System.Windows.Forms.RadioButton();
            this.rbWu = new System.Windows.Forms.RadioButton();
            this.btnClear = new System.Windows.Forms.Button();
            this.infoLabel = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.canvasPictureBox)).BeginInit();
            this.panelRight.SuspendLayout();
            this.SuspendLayout();
            // 
            // canvasPictureBox
            // 
            this.canvasPictureBox.BackColor = System.Drawing.Color.White;
            this.canvasPictureBox.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.canvasPictureBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.canvasPictureBox.Location = new System.Drawing.Point(0, 0);
            this.canvasPictureBox.Name = "canvasPictureBox";
            this.canvasPictureBox.Size = new System.Drawing.Size(584, 561);
            this.canvasPictureBox.TabIndex = 0;
            this.canvasPictureBox.TabStop = false;
            this.canvasPictureBox.MouseDown += new System.Windows.Forms.MouseEventHandler(this.canvasPictureBox_MouseDown);
            // 
            // panelRight
            // 
            this.panelRight.Controls.Add(this.infoLabel);
            this.panelRight.Controls.Add(this.btnClear);
            this.panelRight.Controls.Add(this.rbWu);
            this.panelRight.Controls.Add(this.rbBresenham);
            this.panelRight.Dock = System.Windows.Forms.DockStyle.Right;
            this.panelRight.Location = new System.Drawing.Point(584, 0);
            this.panelRight.Name = "panelRight";
            this.panelRight.Size = new System.Drawing.Size(200, 561);
            this.panelRight.TabIndex = 1;
            // 
            // rbBresenham
            // 
            this.rbBresenham.AutoSize = true;
            this.rbBresenham.Checked = true;
            this.rbBresenham.Location = new System.Drawing.Point(12, 12);
            this.rbBresenham.Name = "rbBresenham";
            this.rbBresenham.Size = new System.Drawing.Size(143, 19);
            this.rbBresenham.TabIndex = 0;
            this.rbBresenham.TabStop = true;
            this.rbBresenham.Text = "Брезенхем (целочисл.)";
            this.rbBresenham.UseVisualStyleBackColor = true;
            // 
            // rbWu
            // 
            this.rbWu.AutoSize = true;
            this.rbWu.Location = new System.Drawing.Point(12, 37);
            this.rbWu.Name = "rbWu";
            this.rbWu.Size = new System.Drawing.Size(103, 19);
            this.rbWu.TabIndex = 1;
            this.rbWu.Text = "Алгоритм Ву";
            this.rbWu.UseVisualStyleBackColor = true;
            // 
            // btnClear
            // 
            this.btnClear.Location = new System.Drawing.Point(12, 75);
            this.btnClear.Name = "btnClear";
            this.btnClear.Size = new System.Drawing.Size(170, 30);
            this.btnClear.TabIndex = 2;
            this.btnClear.Text = "Очистить поле";
            this.btnClear.UseVisualStyleBackColor = true;
            this.btnClear.Click += new System.EventHandler(this.btnClear_Click);
            // 
            // infoLabel
            // 
            this.infoLabel.AutoSize = true;
            this.infoLabel.Location = new System.Drawing.Point(12, 120);
            this.infoLabel.Name = "infoLabel";
            this.infoLabel.Size = new System.Drawing.Size(164, 45);
            this.infoLabel.TabIndex = 3;
            this.infoLabel.Text = "Рисование:\r\n1-й клик по полю — начало,\r\n2-й клик — конец отрезка.";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(784, 561);
            this.Controls.Add(this.canvasPictureBox);
            this.Controls.Add(this.panelRight);
            this.MinimumSize = new System.Drawing.Size(600, 400);
            this.Name = "Form1";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Лаба 3 — Брезенхем и Ву";
            ((System.ComponentModel.ISupportInitialize)(this.canvasPictureBox)).EndInit();
            this.panelRight.ResumeLayout(false);
            this.panelRight.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion
    }
}
