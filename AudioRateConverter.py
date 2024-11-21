import sys
import soundfile as sf
import numpy as np
from scipy.signal import iirdesign, sosfiltfilt
from scipy.signal import resample_poly
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QLineEdit, QListWidget, QComboBox,  QMessageBox
)
from mutagen.flac import FLAC
import os


class AudioProcessor:
    def __init__(self, filepath, chunk_size=44100):
        self.filepath = filepath
        self.samplerate = None
        self.metadata = FLAC(filepath)
        self.chunk_size = chunk_size  # 1秒分（例: 44100サンプル）

    def process_in_chunks(self, output_path, cutoff, stopband_atten, filter_type='lowpass', 
                            target_samplerate=None, target_bitdepth=None):

        with sf.SoundFile(self.filepath, mode='r') as infile:
            self.samplerate = infile.samplerate
            
            if infile.subtype == 'PCM_24':
                self.bitdepth = 24
            elif infile.subtype == 'PCM_16':
                self.bitdepth = 16

            wp1 = cutoff                # 通過域遮断周波数[Hz]
            ws1 = cutoff * 1.05         # 阻止域遮断周波数[Hz]
            gpass1 = 0.5                # 通過域最大損失量[dB]
            gstop1 = stopband_atten     # 阻止域最小減衰量[dB]
            
            sos = iirdesign(wp1, ws1, gpass1, gstop1, output='sos', ftype='cheby2', fs=target_samplerate)

            # 出力のサンプルレートを設定
            output_samplerate = target_samplerate if target_samplerate else self.samplerate

            with sf.SoundFile(output_path, mode='w',
                            samplerate=output_samplerate,
                            channels=infile.channels,
                            subtype='PCM_24' if target_bitdepth == 24 else 'PCM_16',
                            format='FLAC') as outfile:
                while True:
                    # チャンク単位でデータを読み取る
                    data = infile.read(self.chunk_size)
                    if not len(data):
                        break
                    
                    # フィルタするデータの箱を宣言
                    filtered_data = data

                    # リサンプリング（必要な場合のみ）
                    if target_samplerate and target_samplerate != self.samplerate:
                        filtered_data = self.resample(filtered_data, self.samplerate, target_samplerate)
                    
                    # フィルタリング
                    filtered_data = sosfiltfilt(sos, filtered_data, axis=0)
                    
                    # 処理済みデータを書き込む
                    outfile.write(filtered_data)
        # メタデータの保存
        self._save_metadata(output_path)

    def resample(self, data, current_rate, target_rate):
        # 分母と分子を用いた整数比を計算
        gcd = np.gcd(int(current_rate), int(target_rate))
        up = int(target_rate // gcd)  # 分子
        down = int(current_rate // gcd)  # 分母

        if data.ndim == 1:  # モノラル
            return resample_poly(data, up, down)
        elif data.ndim == 2:  # ステレオや多チャンネル
            # 各チャンネルごとにリサンプリング
            resampled = [resample_poly(data[:, ch], up, down) for ch in range(data.shape[1])]
            return np.column_stack(resampled)

    def _save_metadata(self, output_path):
        new_metadata = FLAC(output_path)
        for key, value in self.metadata.tags.items():
            new_metadata[key] = value
        new_metadata.save()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("AudioRateConverter with Filter")
        self.layout = QVBoxLayout()

        self.file_list_label = QLabel("Selected Files:")
        self.layout.addWidget(self.file_list_label)

        self.file_list_widget = QListWidget()
        self.layout.addWidget(self.file_list_widget)

        self.select_files_button = QPushButton("Add FLAC Files")
        self.select_files_button.clicked.connect(self.select_files)
        self.layout.addWidget(self.select_files_button)
        
        self.output_directory_label = QLabel("Output Directory")
        self.layout.addWidget(self.output_directory_label)
        
        self.output_directory = QLineEdit()
        self.layout.addWidget(self.output_directory)
        
        self.select_directory_button = QPushButton("Get Output Directory")
        self.select_directory_button.clicked.connect(self.select_folder)
        self.layout.addWidget(self.select_directory_button)

        self.filter_cutoff_label = QLabel("Filter Cutoff Frequency (Hz):")
        self.layout.addWidget(self.filter_cutoff_label)

        self.filter_cutoff_input = QLineEdit("22000")
        self.layout.addWidget(self.filter_cutoff_input)

        self.filter_stopband_label = QLabel("Filter stop band attenuation (dB):")
        self.layout.addWidget(self.filter_stopband_label)
        
        self.filter_stopband_atten = QLineEdit("150")
        self.layout.addWidget(self.filter_stopband_atten)

        self.resample_rate_label = QLabel("Resample Rate (Hz):")
        self.layout.addWidget(self.resample_rate_label)

        self.resample_rate_input = QComboBox()
        self.resample_rate_input.addItems(["44100", "48000", "88200", "96000", 
                                            "176400", "192000", "352800", "384000"])
        self.layout.addWidget(self.resample_rate_input)
        
        self.bitdepth_label = QLabel("Target Bit Depth:")
        self.layout.addWidget(self.bitdepth_label)
        
        self.bitdepth_input = QComboBox()
        self.bitdepth_input.addItems(["16", "24"])
        self.layout.addWidget(self.bitdepth_input)

        self.process_button = QPushButton("Process All Files")
        self.process_button.clicked.connect(self.process_files)
        self.layout.addWidget(self.process_button)

        self.status_label = QLabel("Status: Waiting for input")
        self.layout.addWidget(self.status_label)

        self.setLayout(self.layout)

    def select_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select FLAC Files", "", "FLAC Files (*.flac)")
        if file_paths:
            for file_path in file_paths:
                self.file_list_widget.addItem(file_path)
    
    def select_folder(self):
        folderpath = QFileDialog.getExistingDirectory(self)
        if folderpath:
            self.output_directory.setText(folderpath)

    def process_files(self):
        try:
            cutoff = float(self.filter_cutoff_input.text())
            resample_rate = int(self.resample_rate_input.currentText())

            for i in range(self.file_list_widget.count()):
                file_path = self.file_list_widget.item(i).text()
                file_name = os.path.basename(file_path)
                output_directory = self.output_directory.text()
                
                if output_directory:
                    output_path = output_directory + "/" +file_name
                else:
                    QMessageBox.warning(self, "Error", "Please specify output directory.")
                    return
                
                bitdepth = int(self.bitdepth_input.currentText())
                stopband_atten = float(self.filter_stopband_atten.text())
                processor = AudioProcessor(file_path)
                processor.process_in_chunks(output_path, cutoff, stopband_atten=stopband_atten, target_samplerate=resample_rate, target_bitdepth=bitdepth)

                self.status_label.setText(f"Processed: {file_path}")

            self.status_label.setText("All files processed successfully.")
        except Exception as e:
            self.status_label.setText(f"Error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
