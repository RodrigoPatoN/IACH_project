"""
This is the main file for the active learning project.
Implements the main loop of the active learning algorithm.
DO NOT DEAL WITH TF MODELS HERE. LAUNCH A SUBPROCESS FOR EACH TASK.
This is due to the device and memory management workarounds
"""
# #HYPERPARAMETERS
# MODEL = 'bert' #model to be used. options: bert, ...
# STRATEGY = 'random' #active learning strategy to be used. options: random, ...
# INITIAL_POOL_SIZE = 0 #initial pool size
# ANNOTATION_SIZE = 100 #number of entries to be annotated in each iteration
# MAX_POOL_SIZE = 1000 #maximum pool size. if not compatible with INITIAL_POOL_SIZE and ANNOTATION_SIZE, the last annotation will be shorter than ANNOTATION_SIZE


from copy import deepcopy
from math import ceil
import os
import numpy as np
from oracle import Oracle
import sklearn.model_selection as sk
import pandas as pd
import subprocess
import time
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='lstm', help='model to be used. options: bert, ...')
    parser.add_argument('-s', '--strategy', type=str, default='random', help='active learning strategy to be used. options: random, ...')
    parser.add_argument('-i', '--initial_pool_size', type=int, default=0, help='initial pool size')
    parser.add_argument('-a', '--annotation_size', type=int, default=100, help='number of entries to be annotated in each iteration')
    parser.add_argument('-f', '--final_pool_size', type=int, default=1000, help='maximum pool size. if not compatible with INITIAL_POOL_SIZE and ANNOTATION_SIZE, the last annotation will be shorter than ANNOTATION_SIZE')
    parser.add_argument('-r', '--resume', action='store_true', help='resume from last run, by loading the datasets in temp_train_dataset.csv and temp_test_dataset.csv')

    args = parser.parse_args()
    return args.model, args.strategy, args.initial_pool_size, args.annotation_size, args.final_pool_size, args.resume


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
    #parse args
    MODEL, STRATEGY, INITIAL_POOL_SIZE, ANNOTATION_SIZE, MAX_POOL_SIZE, RESUME = parse_args()

    #initialize run
    print('initializing run')
    run_nonce = 10000000000 - int(time.time()) #this is a nonce to be used in the run name. subtraction because this way the most recent is at the top of the list
    run_name = f'run_{run_nonce}_{MODEL}_{STRATEGY}'
    print(f'run name: {run_name}')
    os.mkdir(f'./results/{run_name}')
    os.mkdir(f'./models/{run_name}')
    placeholder_results = pd.read_csv('./results/expected_results.csv', comment='#')
    placeholder_results.to_csv(f'./results/{run_name}/results.csv', index=False)
    #initialize temporary models as the initial model, unless resume is set to true
    initial_model_name = f'initial_{MODEL}.keras'
    temp_model_name = f'temp_{MODEL}.keras'
    
    if not RESUME:
        initial_model_name = f'initial_{MODEL}.keras'
        temp_model_name = f'temp_{MODEL}.keras'
        shutil.copyfile(f'./models/{initial_model_name}', f'./temp/{temp_model_name}')





    #initialize datasets
    
    train_dataset = pd.read_csv("./datasets/train_data.csv", on_bad_lines="skip", index_col=0)
    test_dataset = pd.read_csv("./datasets/test_data.csv", on_bad_lines="skip", index_col=0)
    test_dataset.to_csv('./temp/temp_test_dataset.csv')
    oracle = Oracle(train_dataset, ['SentimentText'], ['Sentiment'])

    if RESUME: #if resuming from last run, load the datasets from the temp files. this is useful if we want annotation sizes different between 0-1000 annotations and 1000-10000 annotations, for example. we can run the first part with 1000 annotations and then resume starting from the same 1000 annotations, but with a different annotation size 
        print('resuming from last run')
        train_dataset_checkpoint = pd.read_csv("./temp/temp_train_dataset.csv", on_bad_lines="skip", index_col=0)
        x, y = oracle.annotate_from_checkpoint(train_dataset_checkpoint)
        if oracle.get_annotated_size() != INITIAL_POOL_SIZE:
            print(f'ERROR: INITIAL_POOL_SIZE is set to {INITIAL_POOL_SIZE} but the checkpoint has {oracle.get_annotated_size()} entries.')
            exit(1)
    print('dataset loaded to oracle')


    
    
    

    #main active learning loop

    n_iterations = ceil((MAX_POOL_SIZE - INITIAL_POOL_SIZE) / ANNOTATION_SIZE) + 1
    start = 1 if RESUME else 0
    for i in range(start, n_iterations):
        if i == 0:
            annotation_size = INITIAL_POOL_SIZE
            #initialize training pool
            print('initializing training pool')
        elif i == n_iterations - 1:
            annotation_size = MAX_POOL_SIZE - oracle.get_annotated_size()
        else:
            annotation_size = ANNOTATION_SIZE

        #annotate the pool
        if annotation_size > 0:
            print(f'selecting {annotation_size} entries for annotation')
            if STRATEGY == 'random':
                attributes, labels = oracle.random_pick(annotation_size)

            elif STRATEGY == 'uncertainty':
                #save the non annotated data to a temp file
                non_annotated_data = oracle.get_non_annotated_data()
                if MODEL == 'bert':
                    non_annotated_data = non_annotated_data.sample(10000) #bert can't handle the whole dataset at once, too slow
                non_annotated_data.to_csv('./temp/temp_predict_data.csv')

                #predict the labels for the non annotated data
                if MODEL == 'bert':
                    launch_subprocess_and_wait(" ".join(['python', 'predict_bert.py', './temp/temp_model.keras', './temp/temp_predict_data.csv', './temp/temp_predictions.csv']))
                elif MODEL == 'lstm':
                    launch_subprocess_and_wait(" ".join(['python', 'predict_lstm.py', './temp/temp_model.keras', './temp/temp_predict_data.csv', './temp/temp_predictions.csv']))
                else:
                    print(f'{MODEL} model is not implemented. go to main.py line 130 to implement it')
                    exit(1)

                preds = pd.read_csv('./temp/temp_predictions.csv', comment='#', index_col=0)
                #pick the #annotation_size entries with the highest uncertainty, basically the ones closer to 0.5
                preds['prediction'] = np.abs(preds['prediction'].to_numpy() - 0.5) #convert to uncertainty
                preds.sort_values(by='prediction', inplace=True)
                idx_to_annotate = preds.index[:annotation_size]
                attributes, labels = oracle.annotate(idx_to_annotate)
                # print(attributes)

            else:
                print(f'{STRATEGY} strategy is not implemented')
                return
        
        
        print(f'Iteration {i+1}/{n_iterations}\nSize of training pool: {oracle.get_annotated_size()}')

        
        
        oracle.to_csv('./temp/temp_train_dataset.csv')
        print(f'training pool checkpoint of size {oracle.get_annotated_size()} saved to temp_train_dataset.csv')
        #train the model
        #apparently the training needs to be done in a whole different process. otherwise it will crash due to oom after some iterations. this way it clears the memory after each iteration
        print('retraining the model')
        nonce = int(time.time())
        if MODEL == 'bert':
            launch_subprocess_and_wait(" ".join(['python', 'train_bert.py', './temp/temp_train_dataset.csv', './temp/temp_test_dataset.csv', f'./temp/temp_model.keras']))
        elif MODEL == 'lstm':
            launch_subprocess_and_wait(" ".join(['python', 'train_lstm.py', './temp/temp_train_dataset.csv', './temp/temp_test_dataset.csv', f'./temp/temp_model.keras']))
        else:
            print(f'{MODEL} model is not implemented')
            exit(1)

        #read the results from the temp file
        last_results = pd.read_csv('./temp/temp_results.csv', comment='#')
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