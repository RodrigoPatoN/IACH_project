"""
This is the main file for the active learning project.
Implements the main loop of the active learning algorithm.
"""
#HYPERPARAMETERS
STRATEGY = 'random' #active learning strategy to be used. options: random, ...
INITIAL_POOL_SIZE = 1000 #initial pool size
ANNOTATION_SIZE = 100 #number of entries to be annotated in each iteration
MAX_POOL_SIZE = 1000 #maximum pool size. if not compatible with INITIAL_POOL_SIZE and ANNOTATION_SIZE, the last annotation will be shorter than ANNOTATION_SIZE

EPOCHS = 1 #number of epochs to be used in each iteration


from copy import deepcopy
from math import ceil
import os
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

    #initialize run
    print('initializing run')
    run_nonce = 10000000000 - int(time.time()) #this is a nonce to be used in the run name. subtraction because this way the most recent is at the top of the list
    run_name = f'run_{run_nonce}'
    print(f'run name: {run_name}')
    os.mkdir(f'./results/{run_name}')
    os.mkdir(f'./models/{run_name}')
    placeholder_results = pd.read_csv('./results/expected_results.csv', comment='#')
    placeholder_results.to_csv(f'./results/{run_name}/results.csv', index=False)



    #initialize datasets
    train_dataset, test_dataset = sk.train_test_split(pd.read_csv("./datasets/Sentiment Analysis Dataset.csv", on_bad_lines="skip"), test_size=0.2)
    test_dataset = test_dataset.sample(1000)
    test_dataset.to_csv('./datasets/temp_test_dataset.csv', index=False)
    oracle = Oracle(train_dataset, ['SentimentText'], ['Sentiment'])
    print('dataset loaded to oracle')

    
    
    

    #main active learning loop

    n_iterations = ceil((MAX_POOL_SIZE - INITIAL_POOL_SIZE) / ANNOTATION_SIZE) + 1
    rest = (MAX_POOL_SIZE - INITIAL_POOL_SIZE) % ANNOTATION_SIZE
    for i in range(n_iterations):
        if i == n_iterations - 1:
            annotation_size = rest
        elif i == 0:
            annotation_size = INITIAL_POOL_SIZE
            #initialize training pool
            print('initializing training pool')
        else:
            annotation_size = ANNOTATION_SIZE

        #annotate the pool
        if STRATEGY == 'random':
            attributes, labels = oracle.random_pick(annotation_size)
        
        #insert new strategies here

        else:
            print(f'{STRATEGY} strategy is not implemented')
            return
        
        
        print(f'Iteration {i}/{n_iterations}\nSize of training pool: {oracle.get_annotated_size()}')

        
        oracle.to_csv('./datasets/temp_train_dataset.csv')
        #train the model
        #apparently the training needs to be done in a whole different process. otherwise it will crash due to oom after some iterations. this way it clears the memory after each iteration
        print('retraining the model')
        nonce = int(time.time())
        launch_subprocess_and_wait(" ".join(['python', 'train_bert.py', './datasets/temp_train_dataset.csv', './datasets/temp_test_dataset.csv', f'./models/bert_{nonce}.h5', str(EPOCHS), str(32)]))

        #read the results from the temp file
        last_results = pd.read_csv('./results/temp_results.csv', comment='#')
        last_results['run_name'] = [run_name for _ in range(len(last_results))] #actually should just be a one value list
        last_results['active_learning_strategy'] = [STRATEGY for _ in range(len(last_results))]
        dataset_size = oracle.get_annotated_size()
        last_results['dataset_size'] = [dataset_size for _ in range(len(last_results))]
        print("appending results to results.csv")
        results = pd.read_csv(f'./results/{run_name}/results.csv', comment='#')
        results = pd.concat([results, last_results], axis=0, ignore_index=True)
        results.to_csv(f'./results/{run_name}/results.csv', index=False)

        print("results appended to results.csv")



if __name__ == '__main__':
    main()