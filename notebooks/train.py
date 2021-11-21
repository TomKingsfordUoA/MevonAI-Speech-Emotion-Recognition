import os

import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Lambda
from tensorflow.keras.models import Model


def process_file(path):
    audio_data, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
    features_mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)

    # pad/truncate length:
    features_mfcc = features_mfcc[:, :216]
    if features_mfcc.shape[1] < 216:
        features_mfcc = np.pad(features_mfcc, ((0, 0), (0, 216 - features_mfcc.shape[1])), mode='constant', constant_values=0)

    assert features_mfcc.shape == (13, 216), features_mfcc.shape
    return features_mfcc


def load_dataset(index_csv):
    df = pd.read_csv(index_csv)
    class_names = list(set(df['emotion']))
    class_weights = compute_class_weight('balanced', classes=class_names, y=df['emotion'])

    df['mfcc'] = pd.Series((process_file(path) for _, path in tqdm.tqdm(df['path'].items(), total=df.shape[0])))
    return df, class_names, class_weights



def main():
    print('Generating features (train)...')
    df_train, class_names, class_weights = load_dataset(os.path.join(os.path.dirname(__file__), '../data/train_ravdess.csv'))

    # Generate model:
    model = Sequential()
    model.add(Conv2D(32, 5,strides=2,padding='same', input_shape=(13,216,1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 5,strides=2,padding='same',))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 5,strides=2,padding='same',))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Flatten())

    model.add(Dense(len(class_names)))
    model.add(Activation('softmax'))

    print(model.summary())

    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )

    X = np.stack([features_mfcc for idx, features_mfcc in df_train['mfcc'].items()])
    y = df_train['emotion'].apply(lambda emotion: class_names.index(emotion)).values
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X.shape)
    print(y.shape)
    print(X_train.shape)
    print(y_train.shape)
    print(X_validation.shape)
    print(y_validation.shape)

    # Train
    checkpoint_name = 'ckpt'
    model.fit(
        X_train,
        y_train,
        batch_size=4, 
        epochs=100, 
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_name, save_weights_only=True, save_best_only=True),
        ],
        validation_data=(X_validation, y_validation),  # TODO(TK): implement validation
        class_weight={idx: weight for idx, weight in enumerate(class_weights)},
    )
    model.load_weights(checkpoint_name)

    # Save:
    model.save(os.path.join(os.path.dirname(__file__), '../mevonai.h5'))


if __name__ == '__main__':
    main()

