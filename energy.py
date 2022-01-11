import pyaudio
import wave
import time
import sys
import librosa
import numpy as np
import joblib
from queue import Queue



def cal_range(x):
    temp = [0, 0, 0, 0]
    peak = []
    for i in abs(x):
        if i > 0 and i < 0.1:
            temp[0] += 1
        elif i >= 0.1 and i < 0.2:
            temp[1] += 1
        elif i >= 0.2 and i < 0.3:
            temp[2] += 1
            peak.append(i)
        elif i >= 0.3 and i < 0.4:
            temp[3] += 1
            peak.append(i)

    if len(peak) == 0:
        avg_peak = 0
    else:
        avg_peak = sum(peak)/len(peak)
    return temp, avg_peak


def feature_extraction(data, sr):
    x = data
    sr = sr
    zero_crossings = librosa.zero_crossings(x, pad=False)
    sum_zerocrossings = sum(zero_crossings)

    de_average = sum(x)/len(x)

    abs_average = sum(abs(x))/len(x)

    range_x, avg_peak = cal_range(x)

    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    spc = spectral_centroids.shape[0]

    return sum_zerocrossings, de_average, abs_average, range_x, avg_peak


def result(x):
    if x == 0:
        return "Monotonic"
    elif x == 1:
        return "Normal"
    else:
        return "Energetic"

if len(sys.argv) < 2:
    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)

# file_path = 'energyVdo\“ฌอน”FBขอโทษปัดหนีแจงตรวจStatement1.3ล.แล้ว.wav'
file_path = sys.argv[1]
model = joblib.load('model\energy_model.pkl')
name = ["Monotonic", "Normal", "Energetic"]
score = [0, 0, 0]

queue = Queue()

wf = wave.open(file_path, 'rb')

window_duration = 3
sr = wf.getframerate()
current_time = 0
runninng_time = 0
chuck_duration = 3
window_samples = int(sr * window_duration)
chuck_sample = int(sr * chuck_duration)
audio_data = np.zeros(window_samples, dtype="int16").astype(np.float32)
# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# define callback (2)


def callback(in_data, frame_count, time_info, status):
    global audio_data, current_time
    data = wf.readframes(frame_count)
    # print(time_info)
    audio_data0 = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    scale = 1./float(1 << ((8 * wf.getsampwidth()) - 1))
    audio_data0 *= scale

    audio_data = np.append(audio_data, audio_data0)
    if len(audio_data) > window_samples:
        audio_data = audio_data[-window_samples :]
        queue.put(audio_data)

    return (data, pyaudio.paContinue)


# open stream using callback (3)
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                stream_callback=callback,
                frames_per_buffer= chuck_sample)

# start the stream (4)
stream.start_stream()

# wait for stream to finish (5)
try:
    j = 1
    while stream.is_active:
        queue_data = queue.get()
        sum_zerocrossings, de_average, abs_average, range_x, avg_peak = feature_extraction(queue_data, sr)
        data = [sum_zerocrossings, de_average, abs_average, range_x[0], range_x[1], range_x[2], range_x[3], avg_peak]
        data_pred = model.predict_proba(np.array([data]))
        confidence_score = max(data_pred[0])
        con_index = np.where(data_pred[0] == confidence_score)[0][0]
        if confidence_score >= 0.95:
            score[con_index] += 1
            prediction = name[con_index]
        else:
            prediction = "Normal"
            score[1] += 1
        print(f'Frame : {j} : Prediction is {prediction}')
        j += 1

except (KeyboardInterrupt, SystemExit):
    # stop stream (6)
    stream.stop_stream()
    stream.close()
    wf.close()

    # close PyAudio (7)
    p.terminate()
