import librosa
import numpy as np

class EnergyExtractor:
    def __init__(self, model):
        self.model = model
        self.name = ["Monotonic", "Normal", "Energetic"]
        self.score = [0, 0, 0] 
        self.framNumber = 1  
        self.predictHistory = []
    
    def cal_range(self, x):
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


    def feature_extraction(self, data, sr):
        x = data
        sr = sr
        zero_crossings = librosa.zero_crossings(x, pad=False)
        sum_zerocrossings = sum(zero_crossings)

        de_average = sum(x)/len(x)

        abs_average = sum(abs(x))/len(x)

        range_x, avg_peak = self.cal_range(x)

        spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
        spc = spectral_centroids.shape[0]

        return sum_zerocrossings, de_average, abs_average, range_x, avg_peak


    def result(self, x):
        if x == 0:
            return "Monotonic"
        elif x == 1:
            return "Normal"
        else:
            return "Energetic"
    
    def extract(self, audio_data, sample_rate):
        sum_zerocrossings, de_average, abs_average, range_x, avg_peak = self.feature_extraction(audio_data, sample_rate)
        data = [sum_zerocrossings, de_average, abs_average, range_x[0], range_x[1], range_x[2], range_x[3], avg_peak]
        data_pred = self.model.predict_proba(np.array([data]))
        confidence_score = max(data_pred[0])
        con_index = np.where(data_pred[0] == confidence_score)[0][0]
        if confidence_score >= 0.95:
            self.score[con_index] += 1
            prediction = self.name[con_index]
        else:
            prediction = "Normal"
            self.score[1] += 1
        print(f'Frame : {self.framNumber} : Prediction is {prediction}')
        self.framNumber += 1
        self.predictHistory.append((self.framNumber, prediction))

        return self.framNumber, prediction