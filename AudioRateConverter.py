import sys
import soundfile as sf
import numpy as np
from scipy.signal import iirdesign, sosfiltfilt, butter, lfilter
from scipy.signal import resample_poly
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QLineEdit, QListWidget, QComboBox,  QMessageBox, QHBoxLayout
)
from mutagen.flac import FLAC
from mutagen.dsf import DSF
import os
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import threading

SIZE_PCM_CHUNK = 10
SIZE_DSD_CHUNK = 8
BYTE_TO_BIT = 8
THREADS = 16

class DSDProcessor:
    def __init__(self, file):
        self.file = file
        
    def _read_dsf_header(self, file):
        # Read the DSF header
        header = file.read(28)
        if header[:4] != b"DSD ":  # Check DSF magic number
            raise ValueError("Invalid DSF file")

        # Extract file size and format chunk size
        file_size = struct.unpack("<Q", header[12:20])[0]
        
        # Skip fmt chunk header
        file.seek(4, 1)
        
        fmt_chunk_size = struct.unpack("<Q", file.read(8))[0]

        # Skip reserved bytes in format chunk
        file.seek(8, 1)

        # Extract format details
        channel_type = struct.unpack("<I", file.read(4))[0]
        self.channel_count = struct.unpack("<I", file.read(4))[0]
        self.dsd_sample_rate = struct.unpack("<I", file.read(4))[0]
        self.dsd_bit_depth = struct.unpack("<I", file.read(4))[0]
        file.seek(8, 1)
        self.block_size_per_ch = struct.unpack("<I", file.read(4))[0]
        
        # Skip to the beginning of the data chunk
        file.seek(4, 1)

        # Read the data chunk header
        data_chunk_header = file.read(12)
        if data_chunk_header[:4] != b"data":  # Check "data" chunk magic number
            raise ValueError("Invalid DSF data chunk")

        # Extract the size of the DSD data
        self.data_size = struct.unpack("<Q", data_chunk_header[4:12])[0]
        self.data_start = file.tell()  # Store the starting position of the data section


    def _process_dsf_chunk(self, dsf_file, offset, lock):
        
        # Total chunk size is 4096 bytes per channel
        total_chunk_size = self.block_size_per_ch * self.channel_count
        with lock:
            dsf_file.seek(offset)
            packed_chunk = dsf_file.read(total_chunk_size)
            if not packed_chunk or len(packed_chunk) < total_chunk_size:
                return None  # End of file or incomplete chunk

        # Split packed data into per-channel arrays
        packed_channels = [
            packed_chunk[i * self.block_size_per_ch:(i + 1) * self.block_size_per_ch]
            for i in range(self.channel_count)
        ]

        # Unpack bits for each channel and store in a list
        unpacked_channels = [
            np.unpackbits(np.frombuffer(ch_data, dtype=np.uint8))
            for ch_data in packed_channels
        ]
        # Stack all channels together
        return np.stack(unpacked_channels)


    def _filter_process_chunk(self, buffer, chunk_data, pcm_sample_rate):
        
        if chunk_data.size == 0:
            return [], []
        
        # Combine buffer and new chunk
        combined_chunk = np.concatenate((buffer, chunk_data), axis=1)
        
        # Convert a chunk of DSD data to PCM
        pcm_data = []
        for ch in range(self.channel_count):
            # Convert 1-bit DSD to signed (-1, 1)
            dsd_signed = (2 * combined_chunk[ch].astype(np.int8) - 1).astype(np.float64)
            
            # Apply low-pass filter
            filtered_data = sosfiltfilt(self.sos, dsd_signed, axis=-1)
            
            # exclude overlap buffer
            filtered_data = filtered_data[buffer.shape[1]:]
            
            # Decimate (downsample)
            # 分母と分子を用いた整数比を計算
            gcd = np.gcd(int(self.dsd_sample_rate), int(pcm_sample_rate))
            up = int(pcm_sample_rate // gcd)  # 分子
            down = int(self.dsd_sample_rate // gcd)  # 分母

            pcm_channel = resample_poly(filtered_data, up, down)
            #decimation_factor = self.dsd_sample_rate // pcm_sample_rate
            #pcm_channel = filtered_data[::decimation_factor]
            pcm_data.append(pcm_channel) # Interleaved PCM data
        
        # Update buffer with the last portion of the current chunk
        overlap_size = buffer.shape[1]
        updated_buffer = combined_chunk[:, -overlap_size:]
        
        return np.stack(pcm_data, axis=1), updated_buffer  
    
    def _dsd_to_pcm_core(self, dsf, in_buffer, offset_bytes, target_chunks, pcm_sample_rate, lock):
        remain_chunks = target_chunks
        offset = offset_bytes
        with ThreadPoolExecutor() as executor:
            futures = []
            while remain_chunks > 0:
                to_read_chunks = min(target_chunks, remain_chunks)
                for i in range(to_read_chunks):
                    future = executor.submit(self._process_dsf_chunk, dsf, offset, lock)
                    futures.append(future)
                    offset += self.channel_count * self.block_size_per_ch
                
                # Process results as they are completed
                out_buffer = np.empty((self.channel_count, 0))
                for future in futures:
                    chunk = future.result()
                    if chunk is None:
                        continue
                    out_buffer = np.concatenate([out_buffer, chunk], 1)
                remain_chunks -= to_read_chunks
            return self._filter_process_chunk(in_buffer, out_buffer, pcm_sample_rate)

    def dsd_to_pcm_stream(self, output_file, pcm_sample_rate, cutoff_freq, stopband_atten):
        
        lock = threading.Lock()  # Lock for thread-safe file access
        
        with open(self.file, "rb") as dsf:
            # Parse the DSF header
            self._read_dsf_header(dsf)
            
            # make a low-pass filter
            wp1 = cutoff_freq               # 通過域遮断周波数[Hz]
            ws1 = pcm_sample_rate / 0.5     # 阻止域遮断周波数[Hz]
            gpass1 = 0.5                    # 通過域最大損失量[dB]
            gstop1 = stopband_atten         # 阻止域最小減衰量[dB]
            
            self.sos = iirdesign(wp1, ws1, gpass1, gstop1, output='sos', ftype='cheby2', fs=self.dsd_sample_rate)
            
            #nyquist_rate = self.dsd_sample_rate / 2
            #self.b, self.a = butter(5, cutoff_freq / nyquist_rate, btype='low')
            
            # Move the file pointer to the start of the data section
            dsf.seek(self.data_start)
            
            # Initialize input buffer for overlap filtering
            in_buffer = np.zeros((self.channel_count, self.block_size_per_ch * BYTE_TO_BIT), dtype=np.uint8)
            
            with sf.SoundFile(output_file, mode='w',
                                samplerate=pcm_sample_rate,
                                channels=self.channel_count,
                                subtype='PCM_24',
                                format='WAV') as wav:
                
                # Thread pool for concurrent processing
                total_chunks = self.data_size // (self.channel_count * self.block_size_per_ch)
                chunk_to_bytes = self.channel_count * self.block_size_per_ch
                with ThreadPoolExecutor() as executor:
                    remain_chunks = total_chunks
                    offset = self.data_start
                    while remain_chunks > 0:
                        futures = []
                        read_chunks = 0
                        for i in range (THREADS):
                            to_read_chunks = min(SIZE_DSD_CHUNK, remain_chunks)
                            future = executor.submit(self._dsd_to_pcm_core, dsf, in_buffer, offset, to_read_chunks, pcm_sample_rate, lock)
                            futures.append(future)
                            offset += to_read_chunks * chunk_to_bytes
                            read_chunks += to_read_chunks

                        for future in futures:
                            pcm_output, in_buffer = future.result()
                            if len(pcm_output) == 0:
                                break
                            wav.write(pcm_output)
                        remain_chunks -= read_chunks
        return 1


class AudioProcessor:
    def __init__(self, filepath, chunk_size=44100*SIZE_PCM_CHUNK):
        self.filepath = filepath
        self.samplerate = None
        self.chunk_size = chunk_size
        
        if self.filepath.endswith(".dsf"):
            self.metadata = DSF(filepath)
        else:
            self.metadata = FLAC(filepath)

    def process_in_chunks(self, output_path, cutoff, stopband_atten, filter_type='lowpass', 
                            target_samplerate=None, target_bitdepth=None):

        # ファイルタイプがFLACのとき
        if self.filepath.endswith(".flac"):
            
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
                        
        # ファイルタイプがFLACでないとき(DSFのとき)
        else:
            pcm_temp_path = output_path.replace(".flac", "_temp.wav")
            dsd_processor = DSDProcessor(self.filepath) 
            
            if not dsd_processor.dsd_to_pcm_stream(pcm_temp_path, target_samplerate, cutoff, stopband_atten):
                QMessageBox.critical(self, "Error", "Failed to convert DSD to PCM.")
                return
            
            with sf.SoundFile(pcm_temp_path, mode='r') as tempfile:
                
                with sf.SoundFile(output_path, mode='w',
                                    samplerate=target_samplerate,
                                    channels=tempfile.channels,
                                    subtype='PCM_24' if target_bitdepth == 24 else 'PCM_16',
                                    format='FLAC') as outfile:
                    while True:
                        # チャンク単位でデータを読み取る
                        data = tempfile.read(self.chunk_size)
                        if not len(data):
                            break
                        
                        # 処理済みデータを書き込む
                        outfile.write(data)

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
        self.layout_0 = QVBoxLayout()

        self.file_list_label = QLabel("Selected Files:")
        self.layout_0.addWidget(self.file_list_label)

        self.file_list_widget = QListWidget()
        self.layout_0.addWidget(self.file_list_widget)

        self.select_files_button = QPushButton("Add FLAC/DSD Files")
        self.select_files_button.clicked.connect(self.select_files)
        self.layout_0.addWidget(self.select_files_button)
        
        self.clear_list_button = QPushButton("Clear File List")
        self.clear_list_button.clicked.connect(self.clear_list)
        self.layout_0.addWidget(self.clear_list_button)
        
        self.output_directory_label = QLabel("Output Directory")
        self.layout_0.addWidget(self.output_directory_label)
        
        self.output_directory = QLineEdit()
        self.layout_0.addWidget(self.output_directory)
        
        self.select_directory_button = QPushButton("Get Output Directory")
        self.select_directory_button.clicked.connect(self.select_folder)
        self.layout_0.addWidget(self.select_directory_button)

        self.filter_cutoff_label = QLabel("Filter Cutoff Frequency (Hz):")
        self.layout_0.addWidget(self.filter_cutoff_label)

        self.filter_cutoff_input = QLineEdit("21000")
        self.layout_0.addWidget(self.filter_cutoff_input)

        self.filter_stopband_label = QLabel("Filter stop band attenuation (dB):")
        self.layout_0.addWidget(self.filter_stopband_label)
        
        self.filter_stopband_atten = QLineEdit("150")
        self.layout_0.addWidget(self.filter_stopband_atten)

        self.resample_rate_label = QLabel("Resample Rate (Hz):")
        self.layout_0.addWidget(self.resample_rate_label)

        self.resample_rate_input = QComboBox()
        self.resample_rate_input.addItems(["44100", "48000", "88200", "96000", 
                                            "176400", "192000", "352800", "384000"])
        self.layout_0.addWidget(self.resample_rate_input)
        
        self.bitdepth_label = QLabel("Target Bit Depth:")
        self.layout_0.addWidget(self.bitdepth_label)
        
        self.bitdepth_input = QComboBox()
        self.bitdepth_input.addItems(["16", "24"])
        self.layout_0.addWidget(self.bitdepth_input)

        self.process_button = QPushButton("Process All Files")
        self.process_button.clicked.connect(self.process_files)
        self.layout_0.addWidget(self.process_button)

        self.status_label = QLabel("Status: Waiting for input")
        self.layout_0.addWidget(self.status_label)

        self.setLayout(self.layout_0)

    def select_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Audio Files", "", "Audio Files (*.flac *.dsf)")
        if file_paths:
            for file_path in file_paths:
                self.file_list_widget.addItem(file_path)
    
    def select_folder(self):
        folderpath = QFileDialog.getExistingDirectory(self)
        if folderpath:
            self.output_directory.setText(folderpath)
    
    def clear_list(self):
        self.file_list_widget.clear()

    def process_files(self):
        try:
            cutoff = float(self.filter_cutoff_input.text())
            resample_rate = int(self.resample_rate_input.currentText())

            for i in range(self.file_list_widget.count()):
                file_path = self.file_list_widget.item(i).text()
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                output_directory = self.output_directory.text()
                
                if output_directory:
                    output_path = output_directory + "/" +file_name + ".flac"
                else:
                    QMessageBox.warning(self, "Error", "Please specify output directory.")
                    return
                
                bitdepth = int(self.bitdepth_input.currentText())
                stopband_atten = float(self.filter_stopband_atten.text())
                processor = AudioProcessor(file_path, resample_rate * SIZE_PCM_CHUNK)
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
