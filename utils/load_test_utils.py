'''
Author: your name
Date: 2021-07-14 10:35:40
LastEditTime: 2021-08-31 20:27:00
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /test_codes/utils/load_test_utils.py
'''

import pickle
import keras
from keras.models import load_model
import autokeras as ak

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys 
sys.path.append('./utils')
from operation_test_utils import modify_model
import argparse
from keras.datasets import mnist,cifar10
import time
import gc
import numpy as np
import copy
import uuid
import csv
import string
import kerastuner
import tensorflow as tf
from kerastuner.engine import hypermodel as hm_module
# from autokeras.engine import compute_gradient as cg # use autokeras env
import pandas as pd

import matplotlib.pyplot as plt  
import autokeras as ak

def get_arch(hyperparameters):
    for key in hyperparameters.values.keys():
        if 'block_type' in key:
            arch=hyperparameters.values[key]
            return arch
    return None

def ol_judge(history,threshold,rate):
    acc=history['accuracy']
    maximum=[]
    minimum=[]
    count=0
    for i in range(len(acc)):
        if i==0 or i ==len(acc)-1:
            continue
        if acc[i]-acc[i-1]>=0 and acc[i]-acc[i+1]>=0:
            maximum.append(acc[i])
        if acc[i]-acc[i-1]<0 and acc[i]-acc[i+1]<0:
            minimum.append(acc[i])
    for i in range(min(len(maximum),len(minimum))):
        if maximum[i]-minimum[i]>=threshold:
            count+=1
    if count>=rate*len(acc):
        return True
    else:
        return False

def has_NaN(output):
    output=np.array(output)
    try:
        result=(np.isnan(output).any() or np.isinf(output).any())
    except:
        result=None
    return result

def max_delta_acc(acc_list):
    if len(acc_list)<=3:
        return 10
    max_delta=0
    for i in range(len(acc_list)-1):
        if acc_list[i+1]-acc_list[i]>max_delta:
            max_delta=acc_list[i+1]-acc_list[i]
    return max_delta

def get_loss(history,unstable_threshold=0.03,unstable_rate=0.2,sc_threshold=0.01):

    train_loss=history['loss']
    train_acc=history['accuracy']
    test_loss=history['val_loss']
    test_acc=history['val_accuracy']
    count=0

    if train_loss!=[]:
        if has_NaN(test_loss) or has_NaN(train_loss) or test_loss[-1]>=1e+5:
            return 'slow_converge'

        if ol_judge(history,unstable_threshold,unstable_rate):  
            return 'oscillating'
        elif max_delta_acc(test_acc)<sc_threshold and max_delta_acc(train_acc)<sc_threshold:
            return 'slow_converge'
        else:
            return 'normal'

def get_modification(input_dict):
    dict_length=len(input_dict.keys())
    output_list=[]
    for i in range(dict_length-1):
        diff_list=[]
        pre_list=input_dict[str(i)]
        next_list=input_dict[str(i+1)]
        for j in range(len(next_list)):
            diff_list.append(next_list[j]-pre_list[j])
        output_list.append(diff_list)
    return output_list

def gradient_norm(gradient_list):
    # assert len(gradient_list)%2==0
    norm_kernel_list=[]
    norm_bias_list=[]
    for i in range(int(len(gradient_list)/2)):
        # average_kernel_list.append(np.mean(np.abs(gradient_list[2*i])))
        # average_bias_list.append(np.mean(np.abs(gradient_list[2*i+1])))
        norm_kernel_list.append(np.linalg.norm(np.array(gradient_list[2*i])))
        # norm_bias_list.append(np.linalg.norm(np.array(gradient_list[2*i+1])))
    return norm_kernel_list#,norm_bias_list

def gradient_zero_radio(gradient_list):
    kernel=[]
    bias=[]
    total_zero=0
    total_size=0
    for i in range(len(gradient_list)):    
        zeros=np.sum(gradient_list[i]==0)
        total_zero+=zeros
        total_size+=gradient_list[i].size
    total=float(total_zero)/float(total_size)
    return total

def gradient_message_summary(gradient_list):
    total_ratio= gradient_zero_radio(gradient_list)

    norm_kernel= gradient_norm(gradient_list)#, norm_bias 
    gra_rate = (norm_kernel[0] / norm_kernel[-1])
    return [norm_kernel,gra_rate], [total_ratio]#, norm_bias

def gradient_issue(gradient_list,threshold_low=1e-3,threshold_low_1=1e-4,threshold_high=70,threshold_die_1=0.7):

    [norm_kernel,gra_rate],[total_ratio]=gradient_message_summary(gradient_list)#avg_bias,
    #[total_ratio,kernel_ratio,bias_ratio,max_zero]\
                    
    for i in range(len(gradient_list)):
        if has_NaN(gradient_list[i]):
            return 'explode'

    if gra_rate<threshold_low and norm_kernel[0]<threshold_low_1:
        return 'vanish'
    elif gra_rate>threshold_high:
        return 'explode'
    elif total_ratio>=threshold_die_1:# or max_zero>=threshold_die_2
        return 'dying'
    # else:
    #     feature_dic['died_relu']=0
    return 'normal'

def get_gradient(gw):
    weight_dict=gw['weight']
    gradient_dict=gw['gradient']

    wgt='normal'
    grad='normal'

    for epoch in weight_dict.keys():
        for i in range(len(weight_dict[epoch])):
            if has_NaN(weight_dict[epoch][i]):
                wgt='nan_weight'
                break
        if wgt=='nan_weight':
            break
    for epoch in gradient_dict.keys():
        grad_result=gradient_issue(gradient_dict[epoch])
        if grad_result!='normal':
            grad=grad_result
            break
    return grad,wgt
    
    

    # weight_modi=get_modification(weight_dict)
    # gradient_modi=get_modification(gradient_dict)
    return grad,wgt
    

def judge_dirs(target_dir):
    params_path=os.path.join(target_dir,'param.pkl')
    gw_path=os.path.join(target_dir,'gradient_weight.pkl')
    his_path=os.path.join(target_dir,'history.pkl')

    with open(params_path, 'rb') as f:#input,bug type,params
        hyperparameters = pickle.load(f)
    with open(his_path, 'rb') as f:#input,bug type,params
        history = pickle.load(f)
    with open(gw_path, 'rb') as f:#input,bug type,params
        gw = pickle.load(f)

    arch=get_arch(hyperparameters)
    loss=get_loss(history)
    grad,wgt=get_gradient(gw)
    

    return arch,loss,grad,wgt

def load_evaluation(algw,evaluation_pkl='./utils/priority_all.pkl'):
    with open(evaluation_pkl, 'rb') as f:#input,bug type,params
        evaluation = pickle.load(f)
    if algw not in evaluation.keys():
        return None,None
    result_dict=evaluation[algw]
    opt_list=list(result_dict.keys())
    for opt in opt_list:
        if result_dict[opt]=='/':
            del result_dict[opt]
    sorted_result = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    operation_list=[r[0] for r in sorted_result]
    with open(os.path.abspath('./utils/priority_pure.pkl'), 'rb') as f:#input,bug type,params
        tmp = pickle.load(f)
    operation_list=tmp[algw]
    return result_dict,operation_list # key: operation+value; value: weight

def read_history_score(trial_dir_list,read_trials=15):
    his_score_list=[]
    for cur_trial in range(read_trials):
        for trial_dir in trial_dir_list:
            if os.path.basename(trial_dir).startswith('{}-'.format(cur_trial)):
                his_pkl=os.path.join(trial_dir,'history.pkl')
                with open(his_pkl, 'rb') as f:#input,bug type,params
                    history = pickle.load(f)
                his_score_list.append(max(history['val_accuracy']))
                break
    return his_score_list

def read_history_whole(trial_dir_list,read_trials=15):
    his_score_list=[]
    log_dict={}
    log_dict['best_score']=0
    log_dict['best_trial']=0
    log_dict['best_time']=None
    for cur_trial in range(read_trials):
        for trial_dir in trial_dir_list:
            if os.path.basename(trial_dir).startswith('{}-'.format(cur_trial)):
                his_pkl=os.path.join(trial_dir,'history.pkl')
                with open(his_pkl, 'rb') as f:#input,bug type,params
                    history = pickle.load(f)
                his_score_list.append(max(history['val_accuracy']))
                if max(history['val_accuracy'])>log_dict['best_score']:
                    log_dict['best_score']=max(history['val_accuracy'])
                    # log_dict['best_trial']=cur_trial
                    # log_dict['best_time']=None
                log_dict[cur_trial]={}
                log_dict[cur_trial]['history']=history
                log_dict[cur_trial]['time']=None
                log_dict[cur_trial]['score']=max(history['val_accuracy'])
                break
    
    # log_dict['best_trial']
    return log_dict

def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'

def plot_line_chart(y_list,label_list,title='Test',x_label='x',y_label='y',save_path='./line_chart.pdf'):
    plt.figure(figsize=(10, 5.5))
    
    x=np.arange(1,len(y_list[0])+1)
    for y in range(len(y_list)):
        l1=plt.plot(x,y_list[y],label=label_list[y])
     
    plt.title(title,fontsize=20)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    # x_stick=np.arange(1,16,2)
    # plt.xticks(x_stick,fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0,0.6)
    # plt.xlim(1,15)
    plt.legend(fontsize=16)#,loc=2
    # use % instead of float
    import matplotlib.ticker as ticker 
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))

    plt.savefig(save_path,dpi=300)

def traversalDir_FirstDir(path):
    tmplist = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file1 in files:
            m = os.path.join(path,file1)
            if (os.path.isdir(m)):
                tmplist.append(m)
                # tmplist1.append(file1)
    return tmplist

def check_move(save_dir):
    if os.path.exists(save_dir):
        dir_list=traversalDir_FirstDir(save_dir)
        if dir_list==[]:
            return None
        num=0
        new_save_dir=None
        for d in dir_list:
            tmp_num=int(os.path.basename(d).split('-')[0])
            if tmp_num>=num:
                new_save_dir=d
                num=tmp_num
        # uuid_name=str(uuid.uuid3(uuid.NAMESPACE_DNS,str(time.time())))[-12:]
        # new_save_dir=os.path.join(new_root_dir,uuid_name)
        # import shutil
        # shutil.move(save_dir,new_save_dir)
        # return new_save_dir
    return new_save_dir

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    return False

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        pass
 
    return False

def get_true_value(value):
    if value=='True':
        return True
    elif value=='False':
        return False
    elif not is_number(value):
        return value
    elif is_int(value):
        return int(value)
    else:
        return float(value)

def special_action(action):
    if action in ['activation','initial']:
        return True
    return False

def write_opt(action,value,write_path='./Test_dir/tmp/tmp_action_value.pkl'):
    opt_dict={}
    opt_dict['action']=action
    opt_dict['value']=value
    with open(os.path.abspath(write_path), 'wb') as f:
        pickle.dump(opt_dict, f)

def read_opt(model,read_path='./Test_dir/tmp/tmp_action_value.pkl'):
    read_path=os.path.abspath(read_path)
    if os.path.exists(read_path):
        with open(read_path, 'rb') as f:#input,bug type,params
            opt_dict = pickle.load(f)
        
        opt=model.optimizer
        for key in model.loss.keys():
            loss=model.loss[key].name
            break

        
        if opt_dict['action']=='activation':
            model=modify_model(model,acti=opt_dict['value'],init=None,method='acti')
        elif opt_dict['action']=='initial':
            model=modify_model(model,acti=None,init=opt_dict['value'],method='init')
        elif isinstance(opt_dict['action'],dict):
            for i in opt_dict['action']:
                if 'activation' in i:
                    model=modify_model(model,acti=opt_dict['value'][i],init=None,method='acti')
                if 'initial' in i:
                    model=modify_model(model,acti=None,init=opt_dict['value'][i],method='init')
        else:
            print('Type Error')
            os._exit(0)
        # os.remove(read_path)
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    return model

def sort_opt_wgt_dict(opt_wgt_dict,opt_list,threshold=0):
    # only output the opts whose weight is over threshold
    new_opt_list=[]
    used_action=[]
    for opt in opt_list:
        if opt_wgt_dict[opt]<0:
            return new_opt_list
        action=opt.split('-')[0]
        if action in used_action:
            continue
        new_opt_list.append(opt)
        used_action.append(action)
        
    return new_opt_list

def write_algw(root_dir):
    import subprocess
    command="/data/zxy/anaconda3/envs/autokeras/bin/python ./utils//get_write_algw.py -d {}" #TODO: your path

    out_path=os.path.join(root_dir,'algw_out')
    out_file = open(out_path, 'w')
    out_file.write('logs\n')
    run_cmd=command.format(root_dir)
    # try:
    # subprocess.call(run_cmd, shell=True, stdout=out_file, stderr=out_file)
    subprocess.Popen(run_cmd, shell=True, stdout=out_file, stderr=out_file)
    pass