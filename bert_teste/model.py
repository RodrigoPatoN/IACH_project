"""
This script contains all the code necessary to deal with the model, including preprocessing, training, etc.
"""

import os
import sys
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import sklearn.model_selection as sk


class BERTModel:
    """
    My version of the bert model , inside a class. Provides methods to deal with training, evaluation, and preprocessing.
    """

    def __init__(self, preprocessor_url, encoder_url):
        """
        Initializes the model, and loads the pretrained bert model from tensorflow hub.
        """
        self.preprocessor = hub.KerasLayer(preprocessor_url)
        self.encoder = hub.KerasLayer(encoder_url)

        # Bert layers
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessed_text = self.preprocessor(text_input)
        outputs = self.encoder(preprocessed_text)

        # Neural network layers
        l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
        l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)
        self.model = tf.keras.Model(inputs=[text_input], outputs=[l])
        self.initial_weights = self.model.get_weights()

    @staticmethod
    def from_file(path):
        """
        Loads a model from a file. Useful for resetting, by loading the initial model.
        """
        return tf.keras.models.load_model(path, custom_objects={'KerasLayer': hub.KerasLayer})

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

    def fit(self, x, y, epochs, batch_size):
        """
        Fits the model
        """
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)

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

    def encode(self, text_sequences: list):
        """
        Encodes a text to the bert embeddings, the ones that will be used by the model.
        :param text_sequences: list of texts to be encoded. MUST BE A LIST, EVEN IF IT'S JUST ONE TEXT_SEQUENCE.
        """
        return self.encoder(self.preprocessor(text_sequences))['pooled_output'].numpy()
    
    def predict(self, text_sequences: list):
        """
        Predicts the label of a text sequence.
        :param text_sequences: list of texts to be encoded. MUST BE A LIST, EVEN IF IT'S JUST ONE TEXT_SEQUENCE.
        """
        return self.model.predict(text_sequences)

        


if __name__ == "__main__":
    dataset = pd.read_csv("./datasets/Sentiment Analysis Dataset.csv", on_bad_lines="skip", nrows=1000)
    train_dataset, test_dataset = sk.train_test_split(dataset, test_size=0.2)
    print(dataset.head())
    x = train_dataset['SentimentText']
    y = train_dataset['Sentiment']
    bert = BERTModel("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
    bert.summary()
    # bert.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(bert.encode(["Hello world, this is a sentence!", "bye world, this is another sentence!"]))
    print(bert.predict(["Hello world, this is a very good sentence!", "bye world, this is another kinda nice sentence!"]))
    # bert.fit(x, y, epochs=15, batch_size=32)
    # x_test = test_dataset['SentimentText']
    # y_test = test_dataset['Sentiment']
    # bert.evaluate(x_test, y_test)
