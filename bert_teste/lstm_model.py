"""
This script contains all the code necessary to deal with the model, including preprocessing, training, etc.
HYPERPARAMETERS ARE DEFINED INSIDE EACH MODEL CLASS.
"""

import os
import sys
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import sklearn.model_selection as sk
import numpy as np




class LSTMModel:
    EPOCHS = 1
    BATCH_SIZE = 32 #for training
    """
    My version of an LSTM model , inside a class. Provides methods to deal with training, evaluation, and preprocessing.
    """

    def __init__(self):
        """
        Initializes the model.
        """
        # Hyperparameters of the model
        self.EMBEDDING_DIM = 768
        self.MAX_LENGTH = 300

        self.start_token = 1
        self.stop_token = 0
        self.vocab_size = 30522
        self.tokenizer = text.BertTokenizer("./bert-base-uncased-vocab.txt", lower_case=True)
        
        # LSTM layers
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.EMBEDDING_DIM, input_length=self.MAX_LENGTH, name='text'),#, mask_zero=True)
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(1, activation='sigmoid', name="output")
        ])
        
        self.initial_weights = self.model.get_weights()


    def summary(self):
        """
        Prints a summary of the model
        """
        self.model.summary()

    def compile(self, optimizer, loss, metrics):
        """
        Compiles the model
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, x, y):
        """
        Fits the model
        """
        x = self.preprocess(x)
        self.model.fit(x, y, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE)

    def evaluate(self, x, y):
        """
        Evaluates the model
        """
        self.model.evaluate(x, y)

    def save(self, path):
        """
        Saves the model
        """
        self.model.save(path)

    #TODO: implement only if decided to go with query by representativity
    # def encode(self, text_sequences: list):
    #     """
    #     Encodes a text to the lstm embeddings, the ones that will be used by the model.
    #     :param text_sequences: list of texts to be encoded. MUST BE A LIST, EVEN IF IT'S JUST ONE TEXT_SEQUENCE.
    #     """
    #     return self.encoder(self.preprocessor(text_sequences))['pooled_output'].numpy()
    
    def predict(self, text_sequences: list, batch_size=16):
        """
        Predicts the label of a text sequence.
        :param text_sequences: list of texts to be encoded. MUST BE A LIST, EVEN IF IT'S JUST ONE TEXT_SEQUENCE.
        """
        text_sequences = self.preprocess(text_sequences)
        return self.model.predict(text_sequences, batch_size=batch_size)
    
    def preprocess(self, text_sequences: list, max_length=300): #max chars (not words) in twitter is 280, no subword tokenizer can generate more than 280+2 tokens, 300 to account for any strange characters in there
        """
        Places the special tokens, tokenizes and pads a text sequence.
        :param text_sequences: list of texts to be encoded. MUST BE A LIST, EVEN IF IT'S JUST ONE TEXT_SEQUENCE.
        """

        text_sequences = self.tokenizer.tokenize(text_sequences)
        text_sequences = text_sequences.merge_dims(-2, -1).numpy()
        text_sequences = tf.keras.utils.pad_sequences(text_sequences, padding='post', maxlen=max_length - 1, value=self.stop_token)
        text_sequences = np.pad(text_sequences, ((0, 0), (1, 0)), constant_values=self.start_token)
        return text_sequences

        



if __name__ == "__main__":
    dataset = pd.read_csv("./datasets/Sentiment Analysis Dataset.csv", on_bad_lines="skip", nrows=1000)
    train_dataset, test_dataset = sk.train_test_split(dataset, test_size=0.2)
    print(dataset.head())
    x = train_dataset['SentimentText']
    y = train_dataset['Sentiment']

    lstm = LSTMModel()
    lstm.compile('adam', 'binary_crossentropy', ['accuracy'])
    lstm.summary()
    print(lstm.preprocess(["hello world", "this is a test, to assure is testable"]), lstm.preprocess(["hello world", "this is a test"]).shape)
    print(lstm.predict(["Hello world, this is a very good sentence!", "bye world, this is another kinda nice sentence!"]))
    lstm.fit(x, y)