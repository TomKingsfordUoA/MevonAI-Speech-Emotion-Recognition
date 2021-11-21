import os
import csv

import tensorflow as tf
import pandas as pd
import numpy as np

from .train import load_dataset


def main():
    # Load model:
    model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../mevonai.h5'))
    print(model.summary())

    # Load data:
    df_train, class_names, class_weights = load_dataset(os.path.join(os.path.dirname(__file__), '../data/train_ravdess.csv'))
    df_test, _, _ = load_dataset(os.path.join(os.path.dirname(__file__), '../data/test_ravdess.csv'))
   
    X_train = np.stack([features_mfcc for idx, features_mfcc in df_train['mfcc'].items()])
    y_train = df_train['emotion'].apply(lambda emotion: class_names.index(emotion)).values

    X_test = np.stack([features_mfcc for idx, features_mfcc in df_test['mfcc'].items()])
    y_test = df_test['emotion'].apply(lambda emotion: class_names.index(emotion)).values

    # Predict:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(y_train.shape)
    print(y_train_pred.shape)
    print(X_train.shape)
    print(df_train.index)

    # Save to CSV:
    df_train_pred = pd.DataFrame(data={**{f'pred_{emotion}': y_train_pred[:, idx] for idx, emotion in enumerate(class_names)}, **{'target': [class_names[idx] for idx in y_train]}}, index=df_train.index)
    df_test_pred = pd.DataFrame(data={**{f'pred_{emotion}': y_test_pred[:, idx] for idx, emotion in enumerate(class_names)}, **{'target': [class_names[idx] for idx in y_test]}}, index=df_test.index)
    df_train_pred.to_csv('data/train_pred.csv', index=True, header=True, quoting=csv.QUOTE_NONNUMERIC)
    df_test_pred.to_csv('data/test_pred.csv', index=True, header=True, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':
    main()

