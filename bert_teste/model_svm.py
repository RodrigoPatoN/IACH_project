"""
This script contains all the code necessary to deal with the model, including preprocessing, training, etc.
HYPERPARAMETERS ARE DEFINED INSIDE EACH MODEL CLASS.
"""

import numpy as np
from sklearn.svm import SVC
import pickle
from transformers import AutoTokenizer
from sklearn.exceptions import NotFittedError



class SVMModel:
    C = 10.0
    KERNEL = 'rbf'
    GAMMA = 'scale'
    """
    My version of an SVM (SVC) model , inside a class. Provides methods to deal with training, evaluation, and preprocessing.
    """

    def __init__(self):
        """
        Initializes the model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained('philschmid/BERT-tweet-eval-emotion')
        self.model = SVC(C=self.C, kernel=self.KERNEL, gamma=self.GAMMA, verbose=2)#, n_jobs=-1)

    @staticmethod
    def from_file(path): #this is a constructor
        """
        Loads a model from a file.
        """

        #instantiates the class
        svm = SVMModel()
        with open(path, 'rb') as f:
            svm.model = pickle.load(f)
        return svm

    def fit(self, x, y):
        """
        Fits the model
        """
        x = self.preprocess(x)
        self.model.fit(x, y)

    def save(self, path):
        """
        Saves the model
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    
    def predict(self, text_sequences: list):
        """
        Predicts the label of a text sequence.
        :param text_sequences: list of texts to be encoded. MUST BE A LIST, EVEN IF IT'S JUST ONE TEXT_SEQUENCE.
        """
        try:
            text_sequences = self.preprocess(text_sequences)
            out = self.model.predict(text_sequences)
        except NotFittedError: #if the model is not fitted, return random predictions
            out = np.random.randint(0, 2, len(text_sequences))

        #reshape to (n, 1) instead of (n,)
        return out.reshape(-1, 1)
    
    def preprocess(self, text_sequences: list, max_length=300): #max chars (not words) in twitter is 280, no subword tokenizer can generate more than 280+2 tokens, 300 to account for any strange characters in there
        """
        Places the special tokens, tokenizes and pads a text sequence.
        :param text_sequences: list of texts to be encoded. MUST BE A LIST, EVEN IF IT'S JUST ONE TEXT_SEQUENCE.
        """
        encoded_sequences = np.array([self.tokenizer.encode(sequence, padding="max_length", truncation="only_first", max_length=max_length) for sequence in text_sequences])
        
        return encoded_sequences

        



if __name__ == "__main__":
    import pandas as pd
    import sklearn.model_selection as sk
    dataset = pd.read_csv("./datasets/Sentiment Analysis Dataset.csv", on_bad_lines="skip", nrows=1000)
    train_dataset, test_dataset = sk.train_test_split(dataset, test_size=0.2)
    print(dataset.head())
    x = train_dataset['SentimentText']
    y = train_dataset['Sentiment']

    svm = SVMModel()
    print(svm.preprocess(["hello world", "this is a test, to assure is testable"]), svm.preprocess(["hello world", "this is a test"]).shape)
    print(svm.predict(["Hello world, this is a very good sentence!", "bye world, this is another kinda nice sentence!"]))
    svm.fit(x, y)
    print(svm.predict(["Hello world, this is a very good sentence!", "bye world, this is another kinda nice sentence!"]))
