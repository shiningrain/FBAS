'''
Author: your name
Date: 2021-08-31 17:13:48
LastEditTime: 2021-08-31 22:23:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /FBAS/demo.py
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('./utils')
from load_test_utils import traversalDir_FirstDir
from utils_data import *
from keras.models import load_model
from keras import backend as K
import pandas
import keras
import sys
import pickle
import tensorflow as tf
import time
import argparse
import autokeras as ak
import shutil

def find_origin(root_dir):
    dir_list=traversalDir_FirstDir(root_dir)
    for trial_dir in dir_list:
        if os.path.basename(trial_dir).startswith('0-'):
            return trial_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test image classification')
    parser.add_argument('--data','-d',default='cifar100',choices=['cifar10','cifar100','stl'], help='dataset')
    parser.add_argument('--origin_path','-op',default='./Test_dir/demo_origin/param.pkl', help='orgin autokeras path')
    parser.add_argument('--result_root_path','-rrp',default='./Test_dir/demo_result', help='orgin autokeras path')
    # parser.add_argument('--result_log_path','-rlp',default='./Test_dir/demo_result/log.pkl', help='orgin autokeras path')
    # parser.add_argument('--trial_num_path','-tnp',default='/data/zxy/DL_autokeras/1Autokeras/test_codes/experiment/greedy_cifar/num.pkl',help='num pkl path')
    parser.add_argument('--epoch','-ep',default=2, help='training epoch')
    parser.add_argument('--trials','-tr',default=20, help='searching trials')
    
    args = parser.parse_args()


    root_path=args.result_root_path
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    else:
        os.makedirs(root_path)
        # origin_path=args.origin_path
        # with open(origin_path,'w') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        #     f.write(autokeras_origin_path)
        #     f.close()
    log_path=os.path.join(root_path,'log.pkl')


    if args.data=='cifar10':
        (x_train, y_train), (x_test, y_test)=cifar10_load_data()
    elif args.data=='cifar100':
        (x_train, y_train), (x_test, y_test)=cifar100_load_data()
    elif args.data=='stl':
        (x_train, y_train), (x_test, y_test)=stl_load_data()
    # for i in range(5):
    start_time=time.time()

    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(log_path):
        log_dict={}
        log_dict['cur_trial']=-1
        log_dict['start_time']=time.time()
        log_dict['param_path']=os.path.abspath(args.origin_path)

        with open(log_path, 'wb') as f:
            pickle.dump(log_dict, f)



    clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=args.trials,tuner='greedy')#,tuner='bayesian'
    # Feed the image classifier with training data.
    clf.fit(x_train, y_train, epochs=args.epoch)



    print('finish')