from src.prepare_fuction import get_spectrogram
import tensorflow as tf
import tensorflow_io as tfio
from pydub import AudioSegment

model = tf.keras.models.load_model('model\model_thai_systhesis0.h5')
class ThaiModel:
  def __init__(self, waveform_slide, waveform_rate, commands = ['กู', 'ควาย', 'คัวย', 'ดอก', 'มึง', 'สัต', 'เสือก', 'หี', 'เหี้ย']):
    self.waveform_slide = waveform_slide
    self.waveform_rate = waveform_rate
    self.commands = commands
    self.result = []
    self.collasp_val = []
    self.rude_detect = {}

  def predict(self, startTime_func = min, endTime_func = max):
    self.result.clear()
    self.collasp_val.clear()
    self.rude_detect.clear()
    for i in range(len(self.waveform_slide)):
      spectrogram = get_spectrogram(self.waveform_slide[i][0])
      spectrogram = tf.expand_dims(spectrogram, 0, name=None)
      prediction = model(spectrogram)
      self.result.append([tf.nn.softmax(prediction[0]), self.waveform_slide[i][1], self.waveform_slide[i][2]])
    return self.timestamp_start(startTime_func, endTime_func)
  
  def collapsing(self, startTime_func, endTime_func):
    for key, values in self.rude_detect.items():
      values.sort(key = lambda x: x[0], reverse = False)
      self.rude_detect[key] = values
 
    for key, values in self.rude_detect.items():
      prev_end = 0
      prev_start = 0
      index = 0
      for value in values:
        if value[0] < prev_end:
          prev_end = endTime_func(prev_end, value[1])
          prev_start = startTime_func(prev_start, value[0])
          self.rude_detect[key][index] = (prev_start, prev_end, 1)
        else:
          prev_end = value[1]
          prev_start = value[0]
        index += 1
    # print(self.rude_detect)

    for key,values in self.rude_detect.items():
      start = values[0][0]
      end = values[0][1]
      for value in values:
        if start != value[0]:
          self.collasp_val.append((key, start, end))
        start = value[0]
        end = value[1]
      self.collasp_val.append((key, start, end))

  def timestamp_start(self, startTime_func, endTime_func):
    self.check = False
    for i in self.result:
      for j in range(len(i[0])):
        if i[0][j] >= 0.9:
          self.check = True
          word = self.commands[j]
          start_time = round(i[1]/self.waveform_rate,2)
          end_time = round(i[2]/self.waveform_rate,2)

          if word in self.rude_detect:
            current_value = (start_time, end_time, 1)
            self.rude_detect[word].append(current_value)
          else:
            self.rude_detect[word] = [(start_time, end_time, 1)]
    print("Original prediction", self.rude_detect)
    self.collapsing(startTime_func, endTime_func)
    return self.check, self.collasp_val

  def toStartTime_Minute(self):
    collasp_valMin = []
    for values in self.collasp_val:
      collasp_valMin.append((values[0], round(values[1]/60 , 2), round(values[2]/60 , 2)))
    return collasp_valMin

  def split_audioPlayer(self, source_path, des_path = ""):
    path_list = []
    for values in self.collasp_val:
      start_time = values[1]
      end_time = values[2]
      word = values[0]
      start_timeMilli = start_time* 1000 #+ 800 #Works in milliseconds
      end_timeMilli = end_time * 1000 #+ 800
      newAudio = AudioSegment.from_wav(source_path)
      newAudio = newAudio[start_time: end_timeMilli]
      des = f"{des_path}/{word}-{start_timeMilli}-{end_timeMilli}"+".wav"
      newAudio.export(des, format="wav")
      path_list.append(des)
    return path_list
