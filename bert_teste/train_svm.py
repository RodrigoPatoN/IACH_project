"""
Helper script that trains an lstm model, based on the command line parameters it acepts.
It is supposed to be launched as a child process from the main script, guarantees isolation
between runs, which reduces problems with memory leaks caused by tensorflow between runs (cache i believe)
Saves the model using a nonce name provided
Saves metrics to a temporary file. It will be overwritten on each run, so process it between runs.

This model assumes some preprocessing has already took place, so specify the path to the tokenized dataset.

Why does it write to a temp file instead of directly to a designated results file? Seems easier to manage all that logic
inside the parent process, and it also allows for more flexibility, since the parent process can decide what to do with
the results and where to put them.
This way, each main run can have its own output file, that will have the results of all the bert runs it launched. Therefore,
as we will have lots of test runs, we will have several output files, and only merge the ones that are actual runs to the main results file.
"""


import argparse
import os
import sys
from model_svm import SVMModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def main(train_dataset_path, test_dataset_path, output_path):
    import pandas as pd

    output_filename = "temp_results.csv"
    print(f"Saving output to {output_filename}")


    #training
    svm = SVMModel().from_file('./models/initial_svm.keras')
    train_dataset = pd.read_csv(train_dataset_path, on_bad_lines="skip")
    test_dataset = pd.read_csv(test_dataset_path, on_bad_lines="skip")
    if train_dataset.shape[0] > 0:
        svm.fit(train_dataset['SentimentText'], train_dataset['Sentiment'])

    #testing
    with open(os.path.join("./temp/", output_filename), 'w') as f:
        f.write("#This is a temporary file. Will be overwritten on each training run.\n"
                "#It contains all the metrics for the last run, so process it before running again.\n"
                "#Read it with pandas.read_csv('temp_results.out', comment='#')\n")
        preds = svm.predict(test_dataset['SentimentText']).flatten()
        # print(preds.shape)
        # print(test_dataset['Sentiment'].to_numpy().shape)
        y_target = test_dataset['Sentiment'].to_numpy()
        
        #metrics
        accuracy = accuracy_score(y_target, preds)
        precision_avg = precision_score(y_target, preds, average='weighted')
        recall_avg = accuracy #could recalculate but there's no need, it is the same as accuracy. just here to remember myself of that
        f1_avg = f1_score(y_target, preds, average='weighted')

        precision_negative, precision_positive = precision_score(y_target, preds, average=None)
        recall_negative, recall_positive = recall_score(y_target, preds, average=None)
        f1_negative, f1_positive = f1_score(y_target, preds, average=None)
        
        results = pd.DataFrame({
            'accuracy': [accuracy],
            'precision_avg': [precision_avg],
            'recall_avg': [recall_avg],
            'f1_avg': [f1_avg],
            'precision_negative': [precision_negative],
            'precision_positive': [precision_positive],
            'recall_negative': [recall_negative],
            'recall_positive': [recall_positive],
            'f1_negative': [f1_negative],
            'f1_positive': [f1_positive]
        })
        f.write(results.to_csv(index=False))
        
        
        # lstm.evaluate(test_dataset['SentimentText'], test_dataset['Sentiment'])
        #not saving models due to space constraints
        # print(lstm.predict(['hello world this is very good', 'omg that is horrible']))
        svm.save(output_path)
        # tf.keras.backend.clear_session() #clears the session to avoid memory leaks


def main_test():
    main('./temp/temp_train_dataset.csv', './temp/temp_test_dataset.csv', './temp/temp_svm.keras')


if __name__ == "__main__":
    if len(sys.argv) > 1: #running from main.py, with args
        parser = argparse.ArgumentParser(description='Train a bert model and save it with a nonce name')
        parser.add_argument('train_dataset_path', type=str, help='path to the train dataset')
        parser.add_argument('test_dataset_path', type=str, help='path to the test dataset')
        parser.add_argument('output_path', type=str, help='path to the output file')
        args = parser.parse_args()
        main(args.train_dataset_path, args.test_dataset_path, args.output_path)
    else: #testing, running this script directly, no args
        main_test()
