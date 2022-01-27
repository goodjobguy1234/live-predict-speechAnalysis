from tkinter import *
import pyaudio
import wave
import time
import sys
import librosa
import numpy as np
import joblib
from queue import Queue
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import math
import os


sys.path.insert(1, './src/')
from energyExtractor import EnergyExtractor
from rudewordSpot import RudeWordSpot

if len(sys.argv) < 2:
    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)


file_path = sys.argv[1]

energy_model = joblib.load('model\energy_model.pkl')
rudeword_spot_model = tf.keras.models.load_model('model\model_thai_systhesis0.h5')

queue = Queue()
rudeQueue = Queue()
energyQueue = Queue()

wf = wave.open(file_path, 'rb')

window_duration = 0.15
sr = wf.getframerate()
current_time = 0
runninng_time = 0
chuck_duration = 0.15

energy_duration = 3
rude_duration = 0.25

energy_sample = int(sr * energy_duration)
rude_sample = int(sr * rude_duration)
window_samples = int(sr * window_duration)

chuck_sample = int(sr * chuck_duration)
audio_data = np.zeros(window_samples, dtype="int16").astype(np.float32)
rude_data = np.zeros(rude_sample, dtype="int16").astype(np.float32)
energy_data = np.zeros(energy_sample, dtype="int16").astype(np.float32)

begin_time = 0

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# define callback (2)

def plotter(audio_data, sr, energyResult, rudeResult):
    time = np.linspace(
    0, # start
    len(audio_data)/ sr,
    num = len(audio_data)
    )

    # clear plot
    plt.clf()

    #wave
    plt.subplot(3,1,1)
    plt.plot(time, audio_data)

    plt.subplot(2,2,3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    if rudeResult != []:
        showString = ' '.join(str(value[0]) + '\n' for value in rudeResult)
        ax.set_facecolor("lightgreen")
        ax.text(
            x=0.5,
            y=0.5,
            s=showString,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            color="red",
            fontweight="bold",
            fontfamily="Tahoma",
            transform=ax.transAxes
        )
    else:
        ax.set_facecolor("salmon")

    plt.subplot(2,2,4)
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(plt.NullLocator())
    ax2.yaxis.set_major_locator(plt.NullLocator())
    ax2.text(
        x=0.5,
        y=0.5,
        s=energyResult,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=12,
        color="red",
        fontweight="bold",
        transform=ax2.transAxes
    )
    
    # plt.subplot(3,1,2)
    # ax3 = plt.gca()
    #draw ful-lenghth emthy bar
    # ax3.axhline(y=1600, xmin=0.5, xmax=1,
    #        linewidth=6, color='#af0b1e', alpha=0.1)

    #draw filled bar
      
    # shows the plot 
    # in new window
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

def callback(in_data, frame_count, time_info, status):
    global rude_data, current_time, energy_data, audio_data
    data = wf.readframes(frame_count)
    # print(time_info)
    audio_data0 = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    scale = 1./float(1 << ((8 * wf.getsampwidth()) - 1))
    audio_data0 *= scale

    audio_data = np.append(audio_data, audio_data0)
    rude_data = np.append(rude_data, audio_data0)
    energy_data = np.append(energy_data, audio_data0)

    if len(audio_data) > window_samples:
        audio_data = audio_data[-window_samples :]
        queue.put((audio_data, frame_count))
   
    if len(rude_data) > rude_sample:
        rude_data = rude_data[-rude_sample :]
        rudeQueue.put((rude_data, frame_count))
    
    if len(energy_data) > energy_sample:
        energy_data = energy_data[-energy_sample :]
        energyQueue.put((energy_data, frame_count))

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
frames = wf.getnframes()
rate = wf.getframerate()
duration = frames/float(rate)

duration_percent = 0
total_duration_percent = 100

energyExtractor = EnergyExtractor(energy_model)
rudewordSpot = RudeWordSpot(rudeword_spot_model, sr)

# wait for stream to finish (5)
try:
    plt.figure()
      
    # title of the plot
    # plt.title("Sound Wave")
      
    # label of x-axis
    # plt.xlabel("Time")
    plt.ion()
    plt.show()

    while True:
        time.sleep(0.1)
        
        energyResult = 'Not Detect'
        rudeResult = []
        queue_data, frame_size = queue.get()

        if not energyQueue.empty():
            energy_queue_data, _ = energyQueue.get()
            _, energyResult = energyExtractor.extract(energy_queue_data, sr)

        if not rudeQueue.empty():
            rude_queue_data, _ = rudeQueue.get()
            rudewordSpot.preProcess(sr, [0.25], rude_queue_data)
            isFound, rudeResult = rudewordSpot.predict()

        plotter(queue_data, sr, energyResult, rudeResult)
        duration_percent += (frame_size/ float(sr)) / duration * 100
        print(f'Reading: {round(duration_percent,2)} /{total_duration_percent}%')
        print(f'energy: {energyResult}')
        print()
        
        print("rudeword detection")
        print("predict_result (times per  3 sec):", rudeResult)
        print()
        print('-----------')
        print()

except (KeyboardInterrupt, SystemExit):
    # stop stream (6)
    stream.stop_stream()
    stream.close()
    wf.close()

    # close PyAudio (7)
    p.terminate()
