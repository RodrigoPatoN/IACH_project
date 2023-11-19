"""
This is the main file for the active learning project.
Implements the main loop of the active learning algorithm.
"""
#HYPERPARAMETERS
INITIAL_POOL_SIZE = 1000 #initial pool size
ANNOTATION_SIZE = 100 #number of entries to be annotated in each iteration
MAX_POOL_SIZE = 10000 #maximum pool size. if not compatible with INITIAL_POOL_SIZE and ANNOTATION_SIZE, the last annotation will be shorter than ANNOTATION_SIZE

EPOCHS = 1 #number of epochs to be used in each iteration


from copy import deepcopy
from math import ceil
import numpy as np
from oracle import Oracle
import sklearn.model_selection as sk
import pandas as pd
import subprocess
import time

def launch_subprocess_and_wait(command):
    # Run the command and capture the output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

    # Read and print the output line by line in real-time
    for line in process.stdout:
        print(line.strip())

    # Wait for the process to finish
    process.wait()

    # Print the final return code
    print(f"Return Code: {process.returncode}")


def main():
    #initialize datasets
    train_dataset, test_dataset = sk.train_test_split(pd.read_csv("./datasets/Sentiment Analysis Dataset.csv", on_bad_lines="skip"), test_size=0.2)
    test_dataset = test_dataset.sample(1000)
    test_dataset.to_csv('./datasets/temp_test_dataset.csv', index=False)
    oracle = Oracle(train_dataset, ['SentimentText'], ['Sentiment'])
    print('dataset loaded to oracle')

    #initialize training pool
    print('initializing training pool')
    x, y = oracle.random_pick(INITIAL_POOL_SIZE)
    

    #main active learning loop

    n_iterations = ceil((MAX_POOL_SIZE - INITIAL_POOL_SIZE) / ANNOTATION_SIZE)
    for i in range(n_iterations):
        print(f'Iteration {i}/{n_iterations}\nSize of training pool: {len(x)}')
        if i == n_iterations - 1:
            annotation_size = MAX_POOL_SIZE - len(x)
        else:
            annotation_size = ANNOTATION_SIZE

        #annotate the pool
        attributes, labels = oracle.random_pick(annotation_size)
        
        oracle.to_csv('./datasets/temp_train_dataset.csv')
        #train the model
        #apparently the training needs to be done in a whole different process. otherwise it will crash due to oom after some iterations. this way it clears the memory after each iteration
        print('retraining the model')
        nonce = int(time.time())
        launch_subprocess_and_wait(" ".join(['python', 'train_bert.py', './datasets/temp_train_dataset.csv', './datasets/temp_test_dataset.csv', f'./models/bert_{nonce}.h5', str(EPOCHS), str(32)]))



if __name__ == '__main__':
    main()