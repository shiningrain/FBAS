'''
Author: your name
Date: 2021-08-24 22:01:18
LastEditTime: 2021-08-31 20:30:41
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /test_codes/write_algw.py
'''
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pickle
import sys
sys.path.append('./utils')
from load_test_utils import judge_dirs

# print(tensorflow.__version__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute model gradient by cpu')
    parser.add_argument('--dir','-d', help='model path')# 'auto' 'cust'

    args = parser.parse_args()
    
    
    arch,loss,grad,wgt=judge_dirs(args.dir)
    algw="{}-{}-{}-{}".format(arch,loss,grad,wgt)
    algw_path=os.path.join(args.dir,'algw.pkl')
    with open(algw_path, 'wb') as f:
        pickle.dump(algw, f)