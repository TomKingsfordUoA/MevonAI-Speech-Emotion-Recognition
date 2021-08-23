import csv
import os
import sys
import typing

import keras
import librosa
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)))  # TODO(TK): replace this with a correct import when mevonai is a package
import bulkDiarize as bk

default_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model', 'lstm_cnn_rectangular_lowdropout_trainedoncustomdata.h5')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
classes = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']


class EmotionRecognizer:
    def __init__(self, model_file: typing.Optional[str] = None):
        if model_file is not None:
            self._model = keras.models.load_model(model_file)
        else:
            self._model = keras.models.load_model(default_model_path)

        self._classes = ('Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised')

    def predict_proba(
            self,
            audio_data: typing.Any,  # TODO(TK): replace with np.typing.ArrayLike when numpy upgrades to 1.20+ (conditional on TensorFlow support)
            sample_rate: int,
    ) -> typing.Dict[str, float]:

        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        result = np.zeros((13, 216))
        result[:mfccs.shape[0], :mfccs.shape[1]] = mfccs
        temp = np.zeros((1, 13, 216))  # np.expand_dims(result, axis=0)
        temp[0] = result
        t = np.expand_dims(temp, axis=3)
        ans = self._model.predict(t).flatten()
        return {emotion: prob for emotion, prob in zip(self._classes, ans)}


def predict(folder, classes, model):
    solutions = []
    filenames=[]
    for subdir in os.listdir(folder):
        # print(subdir)
        
        lst = []
        predictions=[]
        # print("Sub",subdir)
        filenames.append(subdir)
        for file in os.listdir(f'{folder}{"/"}{subdir}'):
            # print(subdir,"+",file)
            temp = np.zeros((1,13,216))
            X, sample_rate = librosa.load(os.path.join(f'{folder}{"/"}{subdir}{"/"}', file), res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
            result = np.zeros((13,216))
            result[:mfccs.shape[0],:mfccs.shape[1]] = mfccs
            temp[0] = result
            t = np.expand_dims(temp,axis=3)
            ans=model.predict(t).flatten()
            if ans.shape[0] != len(classes):
                raise RuntimeError("Unexpected number of classes encountered")
            # print("SOL",classes[ans[0]])
            predictions.append(classes[np.argmax(ans)])

        if len(predictions) < 2:
            predictions.append('None')    
        solutions.append(predictions)
    return solutions,filenames


if __name__ == '__main__':
    model = keras.models.load_model(default_model_path)
    INPUT_FOLDER_PATH = "input/"
    OUTPUT_FOLDER_PATH = "output/"
    # bk.diarizeFromFolder(INPUT_FOLDER_PATH,OUTPUT_FOLDER_PATH)
    for subdir in os.listdir(INPUT_FOLDER_PATH):
        bk.diarizeFromFolder(f'{INPUT_FOLDER_PATH}{subdir}{"/"}',(f'{OUTPUT_FOLDER_PATH}{subdir}{"/"}'))
        print("Diarized",subdir)

    folder = OUTPUT_FOLDER_PATH
    for subdir in os.listdir(folder):
        predictions,filenames = predict(f'{folder}{"/"}{subdir}', classes, model)
        # print("filename:",filenames,",Predictions:",predictions)
        with open('SER_'+subdir+'.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            for i in range(len(filenames)):
                csvData = [filenames[i], 'person01',predictions[i][0],'person02',predictions[i][1]]
                print("filename:",filenames[i],",Predicted Emotion := Person1:",predictions[i][0],",Person2:",predictions[i][1])
                writer.writerow(csvData)
        csvFile.close()
    os.remove("filterTemp.wav")
