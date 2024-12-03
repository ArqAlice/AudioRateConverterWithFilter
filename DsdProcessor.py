import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait
import soundfile as sf
from scipy.signal import iirdesign, sosfiltfilt, butter, lfilter
from scipy.signal import resample_poly
import struct

SIZE_DSD_CHUNK = 256
FILTER_BUFFER_CHUNKS = 128
BYTE_TO_BIT = 8


def _read_dsf_header(file):
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
    channel_count = struct.unpack("<I", file.read(4))[0]
    dsd_sample_rate = struct.unpack("<I", file.read(4))[0]
    dsd_bit_depth = struct.unpack("<I", file.read(4))[0]
    sample_count = struct.unpack("<Q", file.read(8))[0]
    block_size_per_ch = struct.unpack("<I", file.read(4))[0]
    
    # Skip to the beginning of the data chunk
    file.seek(4, 1)

    # Read the data chunk header
    data_chunk_header = file.read(12)
    if data_chunk_header[:4] != b"data":  # Check "data" chunk magic number
        raise ValueError("Invalid DSF data chunk")
    
    # Store the starting position of the data section
    data_start = file.tell()  
    
    audio_attr = {}
    audio_attr["channel_count"] = channel_count
    audio_attr["block_size_per_ch"] = block_size_per_ch
    audio_attr["data_start"] = data_start
    audio_attr["dsd_sample_rate"] = dsd_sample_rate
    audio_attr["dsd_bit_depth"] = dsd_bit_depth
    audio_attr["data_size"] = sample_count * channel_count / 8

    return audio_attr


def _read_and_pack_dsf_chunk(audio_attr, file, offset):
    # Acquire chunk data from dsf file
    total_chunk_size = audio_attr["block_size_per_ch"] * audio_attr["channel_count"]

    file.seek(offset)
    packed_chunk = np.frombuffer(file.read(total_chunk_size), dtype=np.uint8)
    if not packed_chunk.any or packed_chunk.size < total_chunk_size:
        return None  # End of file or incomplete chunk
    return packed_chunk

def _unpack_dsf_chunk(audio_attr, packed_chunk):
    
    if packed_chunk is None:
        return None
    
    # Split packed data into per-channel arrays
    packed_channels = [
        packed_chunk[i * audio_attr["block_size_per_ch"] : (i + 1) * audio_attr["block_size_per_ch"]]
        for i in range(audio_attr["channel_count"])
    ]
    
    # Unpack bits for each channel and store in a list
    unpacked_channels = np.empty((audio_attr["channel_count"], audio_attr["block_size_per_ch"] * 8))
    for ch in range(audio_attr["channel_count"]):
        unpacked_1chnnel = np.unpackbits(packed_channels[ch]).reshape(-1, 8)
        
        if audio_attr["dsd_bit_depth"] == 1:
            unpacked_channels[ch] = (np.fliplr(unpacked_1chnnel)).flatten()
        else:
            unpacked_channels[ch] = unpacked_1chnnel.flatten()
    
    # Stack all channels together
    return np.stack(unpacked_channels)


def _process_dsf_chunks(audio_attr, dsf, offset_bytes, target_chunks):
    remain_chunks = target_chunks
    offset = offset_bytes
    with ThreadPoolExecutor() as executor:
        while remain_chunks > 0:
            futures = []
            to_read_chunks = min(target_chunks, remain_chunks)
            for i in range(to_read_chunks):
                packed_chunk = _read_and_pack_dsf_chunk(audio_attr, dsf, offset)
                future = executor.submit(_unpack_dsf_chunk, audio_attr, packed_chunk)
                futures.append(future)
                offset += audio_attr["channel_count"] * audio_attr["block_size_per_ch"]
            
            # Process results as they are completed
            out_buffer = np.empty((audio_attr["channel_count"], 0))
            
            # wait for acquiring chunks
            wait(futures)
            
            # if first read, buffer is zeros
            buffer_size = audio_attr["block_size_per_ch"] * FILTER_BUFFER_CHUNKS
            if offset < audio_attr["data_start"] + buffer_size:
                in_buffer = np.zeros((audio_attr["channel_count"], (buffer_size - offset_bytes) * BYTE_TO_BIT), dtype=np.uint8)
            # if not first read, buffer is first chunks
            else:
                in_buffer = np.empty((audio_attr["channel_count"], 0))
                for i in range(FILTER_BUFFER_CHUNKS):
                    result_buf = futures[i].result()
                    if result_buf is None:
                        result_buf = np.zeros((audio_attr["channel_count"], audio_attr["block_size_per_ch"] * BYTE_TO_BIT), dtype=np.uint8)
                    in_buffer = np.concatenate([in_buffer, result_buf], 1)
            
            # packing chunks to convert
            for future in futures:
                chunk = future.result()
                if chunk is None:
                    continue
                out_buffer = np.concatenate([out_buffer, chunk], 1)
            remain_chunks -= to_read_chunks
        return out_buffer, in_buffer

def _convert_to_pcm_raw(audio_attr, lpf_param, buffer, chunk_data, pcm_sample_rate):
    
    if chunk_data is None:
        return None
    
    # Combine buffer and new chunk
    combined_chunk = np.concatenate((buffer, chunk_data), axis=1)
    
    # Convert a chunk of DSD data to PCM
    pcm_data = []
    for ch in range(audio_attr["channel_count"]):
        # Convert 1-bit DSD to signed (-1, 1)
        dsd_signed = (2 * combined_chunk[ch].astype(np.int8) - 1).astype(np.float64)
        
        # Apply low-pass filter
        filtered_data = sosfiltfilt(lpf_param, dsd_signed, axis=0)
        
        # exclude overlap buffer
        filtered_data = filtered_data[buffer.shape[1]:]
        
        # Decimate (downsample)
        # 分母と分子を用いた整数比を計算
        gcd = np.gcd(int(audio_attr["dsd_sample_rate"]), int(pcm_sample_rate))
        up = int(pcm_sample_rate // gcd)  # 分子
        down = int(audio_attr["dsd_sample_rate"] // gcd)  # 分母

        pcm_channel = resample_poly(filtered_data, up, down)
        #decimation_factor = self.dsd_sample_rate // pcm_sample_rate
        #pcm_channel = filtered_data[::decimation_factor]
        pcm_data.append(pcm_channel) # Interleaved PCM data
    
    return np.stack(pcm_data, axis=1)  

def dsd_to_pcm_stream(dsf_file, output_file, threads ,pcm_sample_rate, cutoff_freq, stopband_atten):
    
    with open(dsf_file, "rb") as dsf:
        # Parse the DSF header
        audio_attr = _read_dsf_header(dsf)
        
        # make a low-pass filter
        wp1 = cutoff_freq               # 通過域遮断周波数[Hz]
        ws1 = pcm_sample_rate / 0.5     # 阻止域遮断周波数[Hz]
        gpass1 = 0.5                    # 通過域最大損失量[dB]
        gstop1 = stopband_atten         # 阻止域最小減衰量[dB]
        
        lpf = iirdesign(wp1, ws1, gpass1, gstop1, output='sos', ftype='cheby2', fs=audio_attr["dsd_sample_rate"])
        
        #nyquist_rate = self.dsd_sample_rate / 2
        #self.b, self.a = butter(5, cutoff_freq / nyquist_rate, btype='low')
        
        # Move the file pointer to the start of the data section
        dsf.seek(audio_attr["data_start"])
        
        with sf.SoundFile(output_file, mode='w',
                            samplerate=pcm_sample_rate,
                            channels=audio_attr["channel_count"],
                            subtype='PCM_24',
                            format='WAV') as wav:
            
            # Thread pool for concurrent processing
            total_chunks = audio_attr["data_size"] // (audio_attr["channel_count"] * audio_attr["block_size_per_ch"])
            chunk_to_bytes = audio_attr["channel_count"] * audio_attr["block_size_per_ch"]

            with ThreadPoolExecutor() as executor:
                remain_chunks = total_chunks
                offset = audio_attr["data_start"]
                while remain_chunks > 0:
                    futures = []
                    read_chunks = 0
                    for i in range (threads):
                        to_read_chunks = min(SIZE_DSD_CHUNK, remain_chunks)
                        to_conv_chunks, buffer = _process_dsf_chunks(audio_attr, dsf, offset, to_read_chunks)
                        future = executor.submit(_convert_to_pcm_raw, audio_attr, lpf, buffer, to_conv_chunks, pcm_sample_rate)
                        futures.append(future)
                        offset += to_read_chunks * chunk_to_bytes
                        read_chunks += to_read_chunks

                    for future in futures:
                        pcm_output = future.result()
                        if pcm_output is None:
                            break
                        wav.write(pcm_output)
                    remain_chunks -= read_chunks
    return 1