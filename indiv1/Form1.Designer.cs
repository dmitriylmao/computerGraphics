using System.Drawing;
using System.Windows.Forms;

namespace Lab5_Graham
{
    partial class Form1
    {
        private System.ComponentModel.IContainer components = null;

        private PictureBox canvasPictureBox;
        private Panel panelRight;
        private Button btnBuildHull;
        private Button btnRandom;
        private Button btnClear;
        private Label lblInfo;
        private Label lblStatus;

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
                components.Dispose();
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        private void InitializeComponent()
        {
            this.canvasPictureBox = new System.Windows.Forms.PictureBox();
            this.panelRight = new System.Windows.Forms.Panel();
            this.lblStatus = new System.Windows.Forms.Label();
            this.lblInfo = new System.Windows.Forms.Label();
            this.btnClear = new System.Windows.Forms.Button();
            this.btnRandom = new System.Windows.Forms.Button();
            this.btnBuildHull = new System.Windows.Forms.Button();
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
            this.canvasPictureBox.Paint += new System.Windows.Forms.PaintEventHandler(this.canvasPictureBox_Paint);
            this.canvasPictureBox.MouseDown += new System.Windows.Forms.MouseEventHandler(this.canvasPictureBox_MouseDown);
            // 
            // panelRight
            // 
            this.panelRight.Controls.Add(this.lblStatus);
            this.panelRight.Controls.Add(this.lblInfo);
            this.panelRight.Controls.Add(this.btnClear);
            this.panelRight.Controls.Add(this.btnRandom);
            this.panelRight.Controls.Add(this.btnBuildHull);
            this.panelRight.Dock = System.Windows.Forms.DockStyle.Right;
            this.panelRight.Location = new System.Drawing.Point(584, 0);
            this.panelRight.Name = "panelRight";
            this.panelRight.Size = new System.Drawing.Size(200, 561);
            this.panelRight.TabIndex = 1;
            // 
            // btnBuildHull
            // 
            this.btnBuildHull.Location = new System.Drawing.Point(12, 12);
            this.btnBuildHull.Name = "btnBuildHull";
            this.btnBuildHull.Size = new System.Drawing.Size(176, 35);
            this.btnBuildHull.TabIndex = 0;
            this.btnBuildHull.Text = "Построить оболочку";
            this.btnBuildHull.UseVisualStyleBackColor = true;
            this.btnBuildHull.Click += new System.EventHandler(this.btnBuildHull_Click);
            // 
            // btnRandom
            // 
            this.btnRandom.Location = new System.Drawing.Point(12, 53);
            this.btnRandom.Name = "btnRandom";
            this.btnRandom.Size = new System.Drawing.Size(176, 35);
            this.btnRandom.TabIndex = 1;
            this.btnRandom.Text = "Случайные точки";
            this.btnRandom.UseVisualStyleBackColor = true;
            this.btnRandom.Click += new System.EventHandler(this.btnRandom_Click);
            // 
            // btnClear
            // 
            this.btnClear.Location = new System.Drawing.Point(12, 94);
            this.btnClear.Name = "btnClear";
            this.btnClear.Size = new System.Drawing.Size(176, 35);
            this.btnClear.TabIndex = 2;
            this.btnClear.Text = "Очистить";
            this.btnClear.UseVisualStyleBackColor = true;
            this.btnClear.Click += new System.EventHandler(this.btnClear_Click);
            // 
            // lblInfo
            // 
            this.lblInfo.AutoSize = true;
            this.lblInfo.Location = new System.Drawing.Point(12, 145);
            this.lblInfo.Name = "lblInfo";
            this.lblInfo.Size = new System.Drawing.Size(176, 60);
            this.lblInfo.TabIndex = 3;
            this.lblInfo.Text = "Управление:\r\n- ЛКМ по полю — добавить точку\r\n- \"Случайные\" — сгенерировать точки\r\n- \"Построить\" — метод Грэхема";
            // 
            // lblStatus
            // 
            this.lblStatus.AutoSize = true;
            this.lblStatus.Location = new System.Drawing.Point(12, 220);
            this.lblStatus.Name = "lblStatus";
            this.lblStatus.Size = new System.Drawing.Size(104, 15);
            this.lblStatus.TabIndex = 4;
            this.lblStatus.Text = "Точек: 0, в обол.: 0";
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
            this.Text = "Выпуклая оболочка — метод Грэхема";
            ((System.ComponentModel.ISupportInitialize)(this.canvasPictureBox)).EndInit();
            this.panelRight.ResumeLayout(false);
            this.panelRight.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion
    }
}
