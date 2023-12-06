"""
This script is used to predict the sentiment of a dataset using a trained LSTM model.
It accepts as inputs the model path, the dataset path and the output path.
It outputs a csv file with the predictions.
"""

import argparse
import os
import sys
import pandas as pd
from model_lstm import LSTMModel
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU') #disable gpu, as it causes memory leaks between runs and constant freezes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the trained model")
    parser.add_argument("dataset_path", help="Path to the dataset to be predicted")
    parser.add_argument("output_path", help="Path to the output file")
    args = parser.parse_args()
    return args.model_path, args.dataset_path, args.output_path

def main(model_path, dataset_path, output_path):
    #load model
    lstm = LSTMModel.from_file(model_path)

    #load dataset
    dataset = pd.read_csv(dataset_path, on_bad_lines="skip", index_col=0)
    # dataset = dataset.sample(1000)
    #predict
    # print(lstm.predict(['hello world this is very good', 'omg that is horrible']))
    preds = lstm.predict(dataset['SentimentText'].to_numpy(), batch_size=2048).flatten()

    #save
    dataset['prediction'] = preds
    dataset.to_csv(output_path)

if __name__ == "__main__":
    if len(sys.argv) > 1: #running from main.py, with args
        model_path, dataset_path, output_path = parse_args()
        main(model_path, dataset_path, output_path)
    else: #testing, running this script directly, no args
        main('./temp/temp_model.keras', './temp/temp_predict_data.csv', './temp/temp_predictions.csv')
