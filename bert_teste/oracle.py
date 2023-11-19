"""
This script will simulate a human annotator, a oracle.
Provides a class, Oracle, that will control the dataset and the information disclosed to the model.
For this, it provides several methods to get the data entries, such as random pick, selected pick, etc.
TODO: address memory problems, partial loading, etc...
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


class Oracle:
    """
    This class will control the dataset and the information disclosed to the model.
    """

    def __init__(self, dataset: pd.DataFrame, attribute_names: list, label_names: list, seed=42):
        """
        Initialize the class.
        :param dataset: dataset to be used, must be a pandas DataFrame.
        :param seed: seed to be used.
        """
        self.dataset = dataset.reset_index(drop=True)
        self.seed = seed
        self.random = random.Random(seed)
        self.attribute_names = attribute_names
        self.label_names = label_names
        self.dataset["annotated"] = [False] * len(self.dataset)

    @staticmethod
    def from_csv(path: str, attribute_names: list, label_names: list, **kwargs):
        """
        Load a dataset from a csv file.
        :param path: path to the csv file.
        :param kwargs: arguments to be passed to pandas.read_csv.
        :return: None.
        """
        dataset = pd.read_csv(path, **kwargs)
        dataset = dataset[attribute_names + label_names]
        return Oracle(dataset, attribute_names, label_names)
    
    def get_dataset(self):
        """
        Get the dataset.
        :return: dataset.
        """
        return self.dataset
    
    def annotate(self, idx_list: list):
        """
        Annotate a list of entries.
        :param idx_list: list of indices to be annotated.
        :return: (list of entries' attributes, list of labels). These will be two numpy arrays, so that they can be directly used on any fit() function
        """

        # get the entries
        entries = self.dataset.iloc[idx_list]

        # get the attributes and labels
        attributes = entries[self.attribute_names].values
        labels = entries[self.label_names].values
        self.dataset.loc[idx_list, "annotated"] = True

        return attributes, labels
    
    def random_pick(self, n: int):
        """
        Get a random sample of n entries.
        :param n: number of entries to be returned.
        :return: (list of entries' attributes, list of labels). These will be two numpy arrays, so that they can be directly used on any fit() function
        """
        # get the indices
        idx_list = self.random.sample(range(len(self.dataset)), n)
        return self.annotate(idx_list)
    
    def to_csv(self, path: str, only_annotated=True, **kwargs):
        """
        Saves a csv file with the dataset. Defaults to only saving annotated rows.
        :param path: path to the csv file.
        :param only_annotated: whether to save only annotated rows. Defaults to True.
        :param kwargs: arguments to be passed to pandas.to_csv.
        """

        if only_annotated:
            self.dataset[self.dataset["annotated"]].to_csv(path, columns = self.attribute_names + self.label_names, **kwargs)


if __name__ == '__main__':
    oracle = Oracle.from_csv('./datasets/Sentiment Analysis Dataset.csv', ['SentimentText'], ['Sentiment'], encoding='utf-8', on_bad_lines='skip')
    print('dataset loaded to oracle')
    print(oracle.annotate([0, 1, 2, 3, 4]))
    print('random sample from oracle')
    print(oracle.random_pick(5))

    print('saving dataset')
    oracle.to_csv('./datasets/short_dataset.csv', index=False)
    print('dataset saved')
