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
            this.loadButton = new System.Windows.Forms.Button();
            this.sourcePictureBox = new System.Windows.Forms.PictureBox();
            this.resultPictureBox = new System.Windows.Forms.PictureBox();
            this.saveButton = new System.Windows.Forms.Button();
            this.hueLabel = new System.Windows.Forms.Label();
            this.hueTrackBar = new System.Windows.Forms.TrackBar();
            this.satTrackBar = new System.Windows.Forms.TrackBar();
            this.valTrackBar = new System.Windows.Forms.TrackBar();
            this.satLabel = new System.Windows.Forms.Label();
            this.valLabel = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.sourcePictureBox)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.resultPictureBox)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.hueTrackBar)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.satTrackBar)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.valTrackBar)).BeginInit();
            this.SuspendLayout();
            // 
            // loadButton
            // 
            this.loadButton.Location = new System.Drawing.Point(16, 26);
            this.loadButton.Name = "loadButton";
            this.loadButton.Size = new System.Drawing.Size(75, 23);
            this.loadButton.TabIndex = 0;
            this.loadButton.Text = "открыть";
            this.loadButton.UseVisualStyleBackColor = true;
            this.loadButton.Click += new System.EventHandler(this.loadButton_Click);
            // 
            // sourcePictureBox
            // 
            this.sourcePictureBox.BackColor = System.Drawing.SystemColors.ButtonFace;
            this.sourcePictureBox.Location = new System.Drawing.Point(231, 12);
            this.sourcePictureBox.Name = "sourcePictureBox";
            this.sourcePictureBox.Size = new System.Drawing.Size(585, 602);
            this.sourcePictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.sourcePictureBox.TabIndex = 1;
            this.sourcePictureBox.TabStop = false;
            // 
            // resultPictureBox
            // 
            this.resultPictureBox.BackColor = System.Drawing.SystemColors.Control;
            this.resultPictureBox.Location = new System.Drawing.Point(839, 12);
            this.resultPictureBox.Name = "resultPictureBox";
            this.resultPictureBox.Size = new System.Drawing.Size(585, 602);
            this.resultPictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.resultPictureBox.TabIndex = 2;
            this.resultPictureBox.TabStop = false;
            // 
            // saveButton
            // 
            this.saveButton.Location = new System.Drawing.Point(16, 75);
            this.saveButton.Name = "saveButton";
            this.saveButton.Size = new System.Drawing.Size(75, 23);
            this.saveButton.TabIndex = 3;
            this.saveButton.Text = "сохранить";
            this.saveButton.UseVisualStyleBackColor = true;
            this.saveButton.Click += new System.EventHandler(this.saveButton_Click);
            // 
            // hueLabel
            // 
            this.hueLabel.AutoSize = true;
            this.hueLabel.Location = new System.Drawing.Point(51, 174);
            this.hueLabel.Name = "hueLabel";
            this.hueLabel.Size = new System.Drawing.Size(35, 13);
            this.hueLabel.TabIndex = 4;
            this.hueLabel.Text = "label1";
            // 
            // hueTrackBar
            // 
            this.hueTrackBar.Location = new System.Drawing.Point(16, 190);
            this.hueTrackBar.Maximum = 180;
            this.hueTrackBar.Minimum = -180;
            this.hueTrackBar.Name = "hueTrackBar";
            this.hueTrackBar.Size = new System.Drawing.Size(104, 45);
            this.hueTrackBar.TabIndex = 5;
            this.hueTrackBar.TickFrequency = 30;
            // 
            // satTrackBar
            // 
            this.satTrackBar.Location = new System.Drawing.Point(16, 287);
            this.satTrackBar.Maximum = 100;
            this.satTrackBar.Minimum = -100;
            this.satTrackBar.Name = "satTrackBar";
            this.satTrackBar.Size = new System.Drawing.Size(104, 45);
            this.satTrackBar.TabIndex = 6;
            this.satTrackBar.TickFrequency = 10;
            // 
            // valTrackBar
            // 
            this.valTrackBar.Location = new System.Drawing.Point(16, 373);
            this.valTrackBar.Maximum = 100;
            this.valTrackBar.Minimum = -100;
            this.valTrackBar.Name = "valTrackBar";
            this.valTrackBar.Size = new System.Drawing.Size(104, 45);
            this.valTrackBar.TabIndex = 7;
            this.valTrackBar.TickFrequency = 10;
            // 
            // satLabel
            // 
            this.satLabel.AutoSize = true;
            this.satLabel.Location = new System.Drawing.Point(51, 271);
            this.satLabel.Name = "satLabel";
            this.satLabel.Size = new System.Drawing.Size(35, 13);
            this.satLabel.TabIndex = 8;
            this.satLabel.Text = "label1";
            // 
            // valLabel
            // 
            this.valLabel.AutoSize = true;
            this.valLabel.Location = new System.Drawing.Point(51, 357);
            this.valLabel.Name = "valLabel";
            this.valLabel.Size = new System.Drawing.Size(35, 13);
            this.valLabel.TabIndex = 9;
            this.valLabel.Text = "label2";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1434, 626);
            this.Controls.Add(this.valLabel);
            this.Controls.Add(this.satLabel);
            this.Controls.Add(this.valTrackBar);
            this.Controls.Add(this.satTrackBar);
            this.Controls.Add(this.hueTrackBar);
            this.Controls.Add(this.hueLabel);
            this.Controls.Add(this.saveButton);
            this.Controls.Add(this.resultPictureBox);
            this.Controls.Add(this.sourcePictureBox);
            this.Controls.Add(this.loadButton);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.sourcePictureBox)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.resultPictureBox)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.hueTrackBar)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.satTrackBar)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.valTrackBar)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button loadButton;
        private System.Windows.Forms.PictureBox sourcePictureBox;
        private System.Windows.Forms.PictureBox resultPictureBox;
        private System.Windows.Forms.Button saveButton;
        private System.Windows.Forms.Label hueLabel;
        private System.Windows.Forms.TrackBar hueTrackBar;
        private System.Windows.Forms.TrackBar satTrackBar;
        private System.Windows.Forms.TrackBar valTrackBar;
        private System.Windows.Forms.Label satLabel;
        private System.Windows.Forms.Label valLabel;
    }
}

