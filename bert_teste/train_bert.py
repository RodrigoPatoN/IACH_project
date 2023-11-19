"""Helper script that trains a bert model, based on the "train_dataset file" and evaluates it
Saves it using a nonce name
"""

import argparse



def main(train_dataset_path, test_dataset_path, output_path, epochs, batch_size):
    import pandas as pd
    from model import BERTModel

    bert = BERTModel.from_file('./models/initial_bert.h5')
    bert.summary()
    train_dataset = pd.read_csv(train_dataset_path, on_bad_lines="skip")
    test_dataset = pd.read_csv(test_dataset_path, on_bad_lines="skip")
    bert.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    bert.fit(train_dataset['SentimentText'], train_dataset['Sentiment'], epochs=epochs, batch_size=batch_size)
    bert.evaluate(test_dataset['SentimentText'], test_dataset['Sentiment'])
    bert.save(output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a bert model and save it with a nonce name')
    parser.add_argument('train_dataset_path', type=str, help='path to the train dataset')
    parser.add_argument('test_dataset_path', type=str, help='path to the test dataset')
    parser.add_argument('output_path', type=str, help='path to the output file')
    parser.add_argument('epochs', type=int, help='number of epochs to train', default=1)
    parser.add_argument('batch_size', type=int, help='batch size', default=32)
    args = parser.parse_args()
    main(args.train_dataset_path, args.test_dataset_path, args.output_path, args.epochs, args.batch_size)