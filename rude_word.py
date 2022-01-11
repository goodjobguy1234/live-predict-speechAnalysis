import pyaudio
import wave
import time
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from queue import Queue
from src.prepare_fuction import *
from thai_model import *
import math

file_path = 'vidDownload\เมื่อโอ๊ตปราโมทย์ เผลอไปกดไลฟ์สดความฮาจึงเกิด.wav'
commands = ['กู', 'ควาย', 'คัวย', 'ดอก', 'มึง', 'สัต', 'เสือก', 'หี', 'เหี้ย']

wf = wave.open(file_path, 'rb')
# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2)
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

sr = get_audio_rate(file_path)

chunk_duration = 1
CHUNK = sr * chunk_duration
# read data
data = wf.readframes(CHUNK)

# play stream (3)
while len(data) > 0:
    stream.write(data)
    data = wf.readframes(CHUNK)
    
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    scale = 1./float(1 << ((8 * wf.getsampwidth()) - 1)) # from librosa
    audio *= scale
    
    waveform_slide, waveform_rate = get_waveform_rb(audio, file_path)
    predict_result = ThaiModel(waveform_slide= waveform_slide, waveform_rate= waveform_rate, commands= commands)
    isFound, result_list = predict_result.predict()
    print("predict_result:", result_list)
    
    
    

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()
