import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import math

AUTOTUNE = tf.data.AUTOTUNE
def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  audio = tf.slice(audio, [0,0], [-1, 1])
  return tf.squeeze(audio, [-1])

def get_audio_rate(audio_path):
  return  tfio.audio.AudioIOTensor(audio_path).rate

def hamming(frames,frameSize):
	frames*=np.hamming(frameSize)
	return frames 

def slide_window(waveform, pad_amount):
  result = []
  for window in pad_amount:
    shift = window//2
    splitted_data = tf.signal.frame(waveform, window, shift, pad_end= True, pad_value=0)
    begin = 0
    splitted_data*=np.hamming(window)
    for data in splitted_data:
      result.append([data,begin, begin + window])
      begin += shift
  return result

def calculatePadding(sample_rate, sec):
  metric = []
  for i in sec:
    metric.append(int(math.ceil(i * sample_rate)))
  return metric

def get_waveform_chunks(waveform, waveform_rate):
  position = tfio.audio.trim(waveform, axis=0, epsilon=0.1)
  # slide_metric = [0.25,0.3,0.35,0.45]
  slide_metric = [0.25]
  windows_size = calculatePadding(sample_rate= waveform_rate, sec=slide_metric)
  start = position[0]
  stop = position[1]
  waveform = waveform[start:stop]
  print(2)
  waveform_slide = slide_window(waveform, windows_size)
  return waveform_slide

def get_waveform(file_path):
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  print(f"this is from decode fuction window: ${waveform} / len is: ${len(waveform)}")
  print(waveform.shape)
  waveform_rate = get_audio_rate(file_path).numpy().item()
  position = tfio.audio.trim(waveform, axis=0, epsilon=0.1)
  # slide_metric = [25000,30000,45000,5000]
  # slide_metric = [0.25,0.3,0.35,0.45]
  slide_metric = [0.25]
  windows_size = calculatePadding(sample_rate= waveform_rate, sec=slide_metric)

  start = position[0]
  stop = position[1]
  waveform = waveform[start:stop]

  waveform_slide = slide_window(waveform, windows_size)
  
  return waveform_slide, waveform_rate

def get_waveform_rb(audio, file_path):
  waveform_rate = get_audio_rate(file_path).numpy().item()
  slide_metric = [0.25]
  windows_size = calculatePadding(sample_rate= waveform_rate, sec=slide_metric)
  waveform = audio
  waveform_slide = slide_window(waveform, windows_size)
  return waveform_slide, waveform_rate
  

def get_spectrogram(waveform):
  zero_padding = tf.zeros([20000] - tf.shape(waveform), dtype=tf.float32)

  waveform = tf.cast(waveform, tf.float32)
  
  equal_length = tf.concat([waveform, zero_padding], 0)

  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)

  spectrogram = tf.abs(spectrogram)
  spectrogram = tf.expand_dims(spectrogram, -1)
  return spectrogram

def split_chucks(audio, padding):
  chunks = []
  # chunks = tf.split(audio, len(audio)//padding, axis=0, num=None, name='split')
  splitted_data = tf.signal.frame(audio, padding, padding, pad_end= True, pad_value=0)
  return splitted_data