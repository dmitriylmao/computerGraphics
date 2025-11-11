namespace Lab1
{
    partial class Form1
    {
        /// <summary>
        /// Обязательная переменная конструктора.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Освободить все используемые ресурсы.
        /// </summary>
        /// <param name="disposing">истинно, если управляемый ресурс должен быть удален; иначе ложно.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Код, автоматически созданный конструктором форм Windows

        /// <summary>
        /// Требуемый метод для поддержки конструктора — не изменяйте 
        /// содержимое этого метода с помощью редактора кода.
        /// </summary>
        private void InitializeComponent()
        {
            this.graphPictureBox = new System.Windows.Forms.PictureBox();
            this.functionComboBox = new System.Windows.Forms.ComboBox();
            ((System.ComponentModel.ISupportInitialize)(this.graphPictureBox)).BeginInit();
            this.SuspendLayout();
            // 
            // graphPictureBox
            // 
            this.graphPictureBox.BackColor = System.Drawing.SystemColors.MenuText;
            this.graphPictureBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.graphPictureBox.Location = new System.Drawing.Point(0, 0);
            this.graphPictureBox.Margin = new System.Windows.Forms.Padding(30, 3, 3, 3);
            this.graphPictureBox.Name = "graphPictureBox";
            this.graphPictureBox.Size = new System.Drawing.Size(800, 450);
            this.graphPictureBox.TabIndex = 0;
            this.graphPictureBox.TabStop = false;
            this.graphPictureBox.Click += new System.EventHandler(this.pictureBox1_Click);
            this.graphPictureBox.Paint += new System.Windows.Forms.PaintEventHandler(this.graphPictureBox_Paint);
            // 
            // functionComboBox
            // 
            this.functionComboBox.FormattingEnabled = true;
            this.functionComboBox.Location = new System.Drawing.Point(12, 12);
            this.functionComboBox.Name = "functionComboBox";
            this.functionComboBox.Size = new System.Drawing.Size(65, 21);
            this.functionComboBox.TabIndex = 1;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.functionComboBox);
            this.Controls.Add(this.graphPictureBox);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.graphPictureBox)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox graphPictureBox;
        private System.Windows.Forms.ComboBox functionComboBox;
    }
}

