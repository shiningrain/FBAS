import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('./utils')
from load_test_utils import traversalDir_FirstDir
from utils_data import *
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import pandas
import tensorflow.keras as keras
import sys
import pickle
import tensorflow as tf
import time
import argparse
import autokeras as ak
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test image classification')
    parser.add_argument('--data','-d',default='cifar100',choices=['cifar10','cifar100','stl'], help='dataset')
    parser.add_argument('--origin_path','-op',default='./Test_dir/demo_origin/param.pkl', help='orgin model architecture path')
    parser.add_argument('--result_root_path','-rrp',default='./Test_dir/demo_result', help='the directory to save results')
    parser.add_argument('--epoch','-ep',default=10, help='training epoch')
    parser.add_argument('--trials','-tr',default=15, help='searching trials')
    
    args = parser.parse_args()

    # set the path and generate the directory
    root_path=args.result_root_path
    tmp_dir=os.path.join(root_path,'log.pkl')
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
        shutil.rmtree(tmp_dir)
    os.makedirs(root_path)
    os.makedirs(tmp_dir)
    log_path=os.path.join(root_path,'log.pkl')
    

    # load datasets
    if args.data=='cifar10':
        (x_train, y_train), (x_test, y_test)=cifar10_load_data()
    elif args.data=='cifar100':
        (x_train, y_train), (x_test, y_test)=cifar100_load_data()
    elif args.data=='stl':
        (x_train, y_train), (x_test, y_test)=stl_load_data()

    # initialize the search log
    if not os.path.exists(log_path):
        log_dict={}
        log_dict['cur_trial']=-1
        log_dict['start_time']=time.time()
        log_dict['param_path']=os.path.abspath(args.origin_path)

        with open(log_path, 'wb') as f:
            pickle.dump(log_dict, f)


    # search models, if you have finished the `setup` in readme.md, the greedy
    # method is our feedback-based search method.
    clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=args.trials,tuner='greedy')#,tuner='bayesian'
    # Feed the image classifier with training data.
    clf.fit(x_train, y_train, epochs=args.epoch)

    print('finish')