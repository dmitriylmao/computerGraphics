using System.Windows.Forms;
using System.Drawing;

namespace Lab4_Bezier
{
    partial class Form1
    {
        private System.ComponentModel.IContainer components = null;

        private PictureBox canvasPictureBox;
        private Panel panelRight;
        private RadioButton rbAdd;
        private RadioButton rbMove;
        private RadioButton rbDelete;
        private Button btnClear;
        private Label infoLabel;
        private Label modeLabel;

        /// <summary>
        /// Очистка ресурсов.
        /// </summary>
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
            this.infoLabel = new System.Windows.Forms.Label();
            this.btnClear = new System.Windows.Forms.Button();
            this.rbDelete = new System.Windows.Forms.RadioButton();
            this.rbMove = new System.Windows.Forms.RadioButton();
            this.rbAdd = new System.Windows.Forms.RadioButton();
            this.modeLabel = new System.Windows.Forms.Label();
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
            this.canvasPictureBox.MouseMove += new System.Windows.Forms.MouseEventHandler(this.canvasPictureBox_MouseMove);
            this.canvasPictureBox.MouseUp += new System.Windows.Forms.MouseEventHandler(this.canvasPictureBox_MouseUp);
            // 
            // panelRight
            // 
            this.panelRight.Controls.Add(this.infoLabel);
            this.panelRight.Controls.Add(this.btnClear);
            this.panelRight.Controls.Add(this.rbDelete);
            this.panelRight.Controls.Add(this.rbMove);
            this.panelRight.Controls.Add(this.rbAdd);
            this.panelRight.Controls.Add(this.modeLabel);
            this.panelRight.Dock = System.Windows.Forms.DockStyle.Right;
            this.panelRight.Location = new System.Drawing.Point(584, 0);
            this.panelRight.Name = "panelRight";
            this.panelRight.Size = new System.Drawing.Size(200, 561);
            this.panelRight.TabIndex = 1;
            // 
            // modeLabel
            // 
            this.modeLabel.AutoSize = true;
            this.modeLabel.Location = new System.Drawing.Point(12, 12);
            this.modeLabel.Name = "modeLabel";
            this.modeLabel.Size = new System.Drawing.Size(103, 15);
            this.modeLabel.TabIndex = 0;
            this.modeLabel.Text = "Режим редактир.:";
            // 
            // rbAdd
            // 
            this.rbAdd.AutoSize = true;
            this.rbAdd.Checked = true;
            this.rbAdd.Location = new System.Drawing.Point(15, 35);
            this.rbAdd.Name = "rbAdd";
            this.rbAdd.Size = new System.Drawing.Size(104, 19);
            this.rbAdd.TabIndex = 1;
            this.rbAdd.TabStop = true;
            this.rbAdd.Text = "Добавление";
            this.rbAdd.UseVisualStyleBackColor = true;
            // 
            // rbMove
            // 
            this.rbMove.AutoSize = true;
            this.rbMove.Location = new System.Drawing.Point(15, 60);
            this.rbMove.Name = "rbMove";
            this.rbMove.Size = new System.Drawing.Size(103, 19);
            this.rbMove.TabIndex = 2;
            this.rbMove.Text = "Перемещение";
            this.rbMove.UseVisualStyleBackColor = true;
            // 
            // rbDelete
            // 
            this.rbDelete.AutoSize = true;
            this.rbDelete.Location = new System.Drawing.Point(15, 85);
            this.rbDelete.Name = "rbDelete";
            this.rbDelete.Size = new System.Drawing.Size(74, 19);
            this.rbDelete.TabIndex = 3;
            this.rbDelete.Text = "Удаление";
            this.rbDelete.UseVisualStyleBackColor = true;
            // 
            // btnClear
            // 
            this.btnClear.Location = new System.Drawing.Point(15, 120);
            this.btnClear.Name = "btnClear";
            this.btnClear.Size = new System.Drawing.Size(170, 30);
            this.btnClear.TabIndex = 4;
            this.btnClear.Text = "Очистить точки";
            this.btnClear.UseVisualStyleBackColor = true;
            this.btnClear.Click += new System.EventHandler(this.btnClear_Click);
            // 
            // infoLabel
            // 
            this.infoLabel.AutoSize = true;
            this.infoLabel.Location = new System.Drawing.Point(15, 170);
            this.infoLabel.Name = "infoLabel";
            this.infoLabel.Size = new System.Drawing.Size(174, 75);
            this.infoLabel.TabIndex = 5;
            this.infoLabel.Text = "Управление:\r\n- Режим \"Добавление\":\r\n  ЛКМ — новая точка.\r\n- \"Перемещение\":\r\n  ЛКМ по точке — тянуть.\r\n- \"Удаление\":\r\n  ЛКМ по точке — удалить.";
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
            this.Text = "Лаба 4 — Кубические сплайны Безье";
            ((System.ComponentModel.ISupportInitialize)(this.canvasPictureBox)).EndInit();
            this.panelRight.ResumeLayout(false);
            this.panelRight.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion
    }
}
