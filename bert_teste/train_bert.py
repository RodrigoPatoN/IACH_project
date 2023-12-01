"""Helper script that trains a bert model, based on the "train_dataset file" and evaluates it
Saves it using a nonce name
"""

import argparse
import sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def main(train_dataset_path, test_dataset_path, output_path, epochs, batch_size):
    import pandas as pd
    from model import BERTModel

    output_filename = output_path.split('/')[-1].split('.')[0] + ".out"
    print(f"Saving output to {output_filename}")

    with open("/home/raulsofia/IACH/bert_teste/results/" + output_filename, "w") as f:
        sys.stdout = f
        bert = BERTModel.from_file('./models/initial_bert.h5')
        bert.summary()
        train_dataset = pd.read_csv(train_dataset_path, on_bad_lines="skip")
        test_dataset = pd.read_csv(test_dataset_path, on_bad_lines="skip")
        bert.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        bert.fit(train_dataset['SentimentText'], train_dataset['Sentiment'], epochs=epochs, batch_size=batch_size)

        preds = bert.predict(test_dataset['SentimentText']).flatten().round()
        print(preds.shape)
        print(test_dataset['Sentiment'].to_numpy().shape)
        accuracy = accuracy_score(test_dataset['Sentiment'].to_numpy(), preds)
        f1 = f1_score(test_dataset['Sentiment'].to_numpy(), preds)
        precision = precision_score(test_dataset['Sentiment'].to_numpy(), preds)
        recall = recall_score(test_dataset['Sentiment'].to_numpy(), preds)
        print(f'Accuracy: {accuracy}\nF1: {f1}\nPrecision: {precision}\nRecall: {recall}')
        
        # bert.evaluate(test_dataset['SentimentText'], test_dataset['Sentiment'])
        bert.save(output_path)


def main_test():
    main('./datasets/temp_train_dataset.csv', './datasets/temp_test_dataset.csv', './models/bert_test.h5', 1, 32)


if __name__ == "__main__":
    # main_test()
    parser = argparse.ArgumentParser(description='Train a bert model and save it with a nonce name')
    parser.add_argument('train_dataset_path', type=str, help='path to the train dataset')
    parser.add_argument('test_dataset_path', type=str, help='path to the test dataset')
    parser.add_argument('output_path', type=str, help='path to the output file')
    parser.add_argument('epochs', type=int, help='number of epochs to train', default=1)
    parser.add_argument('batch_size', type=int, help='batch size', default=32)
    args = parser.parse_args()
    main(args.train_dataset_path, args.test_dataset_path, args.output_path, args.epochs, args.batch_size)