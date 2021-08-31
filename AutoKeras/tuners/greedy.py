# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import kerastuner
import numpy as np

from autokeras.engine import tuner as tuner_module


class TrieNode(object):
    def __init__(self):
        super().__init__()
        self.num_leaves = 0
        self.children = {}
        self.hp_name = None

    def is_leaf(self):
        return len(self.children) == 0


class Trie(object):
    def __init__(self):
        super().__init__()
        self.root = TrieNode()

    def insert(self, hp_name):
        names = hp_name.split("/")

        new_word = False
        current_node = self.root
        nodes_on_path = [current_node]
        for name in names:
            if name not in current_node.children:
                current_node.children[name] = TrieNode()
                new_word = True
            current_node = current_node.children[name]
            nodes_on_path.append(current_node)
        current_node.hp_name = hp_name

        if new_word:
            for node in nodes_on_path:
                node.num_leaves += 1

    @property
    def nodes(self):
        return self._get_all_nodes(self.root)

    def _get_all_nodes(self, node):
        ret = [node]
        for key, value in node.children.items():
            ret += self._get_all_nodes(value)
        return ret

    def get_hp_names(self, node):
        if node.is_leaf():
            return [node.hp_name]
        ret = []
        for key, value in node.children.items():
            ret += self.get_hp_names(value)
        return ret


class GreedyOracle(kerastuner.Oracle):
    """An oracle combining random search and greedy algorithm.

    It groups the HyperParameters into several categories, namely, HyperGraph,
    Preprocessor, Architecture, and Optimization. The oracle tunes each group
    separately using random search. In each trial, it use a greedy strategy to
    generate new values for one of the categories of HyperParameters and use the best
    trial so far for the rest of the HyperParameters values.

    # Arguments
        initial_hps: A list of dictionaries in the form of
            {HyperParameter name (String): HyperParameter value}.
            Each dictionary is one set of HyperParameters, which are used as the
            initial trials for the search. Defaults to None.
        seed: Int. Random seed.
    """

    def __init__(self, initial_hps=None, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.initial_hps = initial_hps or []
        self._tried_initial_hps = [False] * len(self.initial_hps)

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                "initial_hps": self.initial_hps,
                "tried_initial_hps": self._tried_initial_hps,
            }
        )
        return state

    def set_state(self, state):
        super().set_state(state)
        self.initial_hps = state["initial_hps"]
        self._tried_initial_hps = state["tried_initial_hps"]

    # zxy
    def get_best_trial_id(self):
        best_trials = self.get_best_trials()
        if best_trials:
            return best_trials[0].trial_id
        else:
            return None

    def get_best_hp_dir(self,save_dir):

        import os
        import pickle
        import sys
        sys.path.append('./utils')
        from load_test_utils import check_move
    
        
        trial_id_path=os.path.join(save_dir,'trial_id.pkl')
        if not os.path.exists(trial_id_path):
            return None
        
        trial_id=self.get_best_trial_id()
        if trial_id==None:
            return check_move(save_dir)

        with open(trial_id_path, 'rb') as f:#input,bug type,params
            trial_id_dict = pickle.load(f)
        return trial_id_dict[trial_id]

    def obtain_new_hps(self,
                        save_dir='./Test_dir/demo_result'):

        import os
        import pickle
        import time
        import sys
        sys.path.append('./utils')
        from load_test_utils import judge_dirs,load_evaluation,check_move,get_true_value,special_action,write_opt,sort_opt_wgt_dict
        start=time.time()
        new_save_dir=self.get_best_hp_dir(save_dir)
        
        # new_save_dir=check_move(save_dir)
        # if no save dir, return None
        if new_save_dir==None:
            return None
        
        # step 1:# read current state and judge the training condition, return
        # algw contidion sign
        
        algw_path=os.path.join(new_save_dir,'algw.pkl')
        if os.path.exists(algw_path):
            with open(algw_path, 'rb') as f:#input,bug type,params
                algw = pickle.load(f)
        else:
            arch,loss,grad,wgt=judge_dirs(new_save_dir)
            algw="{}-{}-{}-{}".format(arch,loss,grad,wgt)
        # step 2:# load evaluation results, if not return None,or get the operation and corresponding weights
        # opt_wgt_dict,opt_list=load_evaluation(algw)
        opt_wgt_dict,opt_list=load_evaluation(algw,evaluation_pkl=os.path.abspath('./utils/priority_all.pkl'))
        time1=time.time()
        print(time1-start)
        if opt_wgt_dict==None:
            return None

        # step 3:
        # opt_list=sort_opt_wgt_dict(opt_wgt_dict,opt_list)#our greedy method
        # values=self.generate_hp_values_greedy(opt_list)
        values=self.generate_hp_values(opt_list)
        print(time.time()-time1)
        print('================We have generated the values! Ready to TRAIN================')
        # with open('/data/zxy/DL_autokeras/1Autokeras/test_codes/experiment/cifar_origin/autokeras_7.20_random_8/best_param.pkl', 'rb') as f:#input,bug type,params
        #     hp = pickle.load(f)
        # values=hp.values
        return values

    def _get_best_action(self,
                        best_hps,
                        operation_list,
                        collision=0,
                        best_hash_path='./Test_dir/demo_result/hash.pkl',
                        method='normal'):

        # zxy

        import os
        import pickle
        import sys
        sys.path.append('./utils')
        from load_test_utils import judge_dirs,load_evaluation,check_move,get_true_value,special_action,write_opt,sort_opt_wgt_dict

        if method=='normal':
            best_hps_key_list=list(best_hps.values.keys())
            best_hps_hash=self._compute_values_hash(best_hps.values)
            if not os.path.exists(best_hash_path):
                best_hash_dict={}
            else:
                with open(best_hash_path, 'rb') as f:#input,bug type,params
                    best_hash_dict = pickle.load(f)
            best_hp_name=None
            best_hp_value=None
            if best_hps_hash not in best_hash_dict.keys():
                for opt in operation_list:
                    if collision>0:
                        collision-=1
                        continue
                    if len(opt.split('-'))==3:#TODO: back
                        continue
                    action=opt.split('-')[0]
                    value=opt.replace('{}-'.format(action), '')
                    if special_action(action):
                        best_hash_dict[best_hps_hash]=[opt]
                        with open(best_hash_path, 'wb') as f:
                            pickle.dump(best_hash_dict, f)
                        return action, value, "Special"

                    # else:# TODO: delete
                    #     print('failed')
                    #     continue

                    for key in best_hps_key_list:
                        if action in key:
                            best_hp_name=key
                            best_hp_value=get_true_value(value)
                            if best_hps[key]==best_hp_value:
                                continue
                            best_hash_dict[best_hps_hash]=[opt]
                            with open(best_hash_path, 'wb') as f:
                                pickle.dump(best_hash_dict, f)
                            
                            return best_hp_name,best_hp_value,None
            else:
                for opt in operation_list:
                    if opt in best_hash_dict[best_hps_hash]:
                        continue
                    if collision>0:
                        collision-=1
                        continue
                    if len(opt.split('-'))==3:#TODO: back
                        continue
                    action=opt.split('-')[0]
                    value=opt.replace('{}-'.format(action), '')
                    if special_action(action):
                        best_hash_dict[best_hps_hash].append(opt)
                        with open(best_hash_path, 'wb') as f:
                            pickle.dump(best_hash_dict, f)
                        return action, value, "Special"
                    for key in best_hps_key_list:
                        if action in key:
                            best_hp_name=key
                            best_hp_value=get_true_value(value)
                            if best_hps[key]==best_hp_value:
                                continue
                            best_hash_dict[best_hps_hash].append(opt)
            
                            with open(best_hash_path, 'wb') as f:
                                pickle.dump(best_hash_dict, f)
                            
                            return best_hp_name,best_hp_value,None
        # elif method=='greedy':
        #     best_hps_key_list=list(best_hps.values.keys())
        #     best_hps_hash=self._compute_values_hash(best_hps.values)
        #     if not os.path.exists(best_hash_path):
        #         best_hash_dict={}
        #     else:
        #         with open(best_hash_path, 'rb') as f:#input,bug type,params
        #             best_hash_dict = pickle.load(f)
        #     best_hp_name=[]
        #     best_hp_value={}
        #     special=None
        #     if best_hps_hash not in best_hash_dict.keys():
        #         best_hash_dict[best_hps_hash]=[]
        #         for opt in operation_list:
        #             if collision>0:
        #                 collision-=1
        #                 continue
        #             action=opt.split('-')[0]
        #             value=opt.replace('{}-'.format(action), '')
        #             if special_action(action):
        #                 best_hash_dict[best_hps_hash].append(opt)
        #                 with open(best_hash_path, 'wb') as f:
        #                     pickle.dump(best_hash_dict, f)
        #                 special="Special"

        #             # else:# TODO: delete
        #             #     print('failed')
        #             #     continue

        #             for key in best_hps_key_list:
        #                 if action in key:
        #                     best_hp_name.append(key)
        #                     best_hp_value[key]=get_true_value(value)
        #                     if best_hps[key]==best_hp_value[key]:
        #                         continue
        #                     best_hash_dict[best_hps_hash].append(opt)
        #                     with open(best_hash_path, 'wb') as f:
        #                         pickle.dump(best_hash_dict, f)
                            
        #         return best_hp_name,best_hp_value,special#TODO: finished here
        #     else:
        #         for opt in operation_list:
        #             if opt in best_hash_dict[best_hps_hash]:
        #                 continue
        #             if collision>0:
        #                 collision-=1
        #                 continue
        #             action=opt.split('-')[0]
        #             value=opt.replace('{}-'.format(action), '')
        #             if special_action(action):
        #                 best_hash_dict[best_hps_hash].append(opt)
        #                 with open(best_hash_path, 'wb') as f:
        #                     pickle.dump(best_hash_dict, f)
        #                 special="Special"
        #             for key in best_hps_key_list:
        #                 if action in key:
        #                     best_hp_name.append(key)
        #                     best_hp_value[key]=get_true_value(value)
        #                     if best_hps[key]==best_hp_value[key]:
        #                         continue
        #                     best_hash_dict[best_hps_hash].append(opt)
            
        #                     with open(best_hash_path, 'wb') as f:
        #                         pickle.dump(best_hash_dict, f)
                            
        #         return best_hp_name,best_hp_value,special

    
    def generate_hp_values(self,operation_list):
        # zxy

        import os
        import pickle
        import random
        import sys
        sys.path.append('./utils')
        from load_test_utils import judge_dirs,load_evaluation,check_move,get_true_value,special_action,write_opt,sort_opt_wgt_dict

        with open(os.path.abspath('./Test_dir/demo_result/log.pkl'), 'rb') as f:#input,bug type,params
            log_dict = pickle.load(f)

        if log_dict['cur_trial']==3 :#or log_dict['cur_trial']==6
            opti=True
            for key in log_dict.keys():
                try:
                    if float(key.split('-')[1])>=0.8:
                        opti=False
                except:
                    print('error')
            if opti:
                optimal_list=['./AutoKeras/utils/optimal_archi/param1.pkl','./AutoKeras/utils/optimal_archi/param2.pkl','./AutoKeras/utils/optimal_archi/param3.pkl','./AutoKeras/utils/optimal_archi/param4.pkl']
                with open(os.path.abspath(random.choice(optimal_list)), 'rb') as f:#input,bug type,params
                    hp = pickle.load(f)
                values=hp.values
                print('===============Use optimal Structure!!===============\n')
                return values

        best_hps = self._get_best_hps()

        collisions = 0
        while True:
            best_hp_name,best_hp_value,special_sign=self._get_best_action(best_hps,operation_list,collisions)
            if special_sign !=None:
                write_opt(best_hp_name,best_hp_value)
                return best_hps.values
            hps = kerastuner.HyperParameters()
            # Generate a set of random values.
            trigger_count=0# if over 1 in generation(error situation), then use greedy
            for hp in self.hyperparameters.space:
                hps.merge([hp])
                # if not active, do nothing.
                # if active, check if selected to be changed.
                if hps.is_active(hp):
                    # if was active and not selected, do nothing.
                    if best_hps.is_active(hp.name) and hp.name != best_hp_name:
                        hps.values[hp.name] = best_hps.values[hp.name]
                        continue
                    # if was not active or selected, sample.
                    hps.values[hp.name] = best_hp_value#hp.random_sample(self._seed_state)
                    trigger_count+=1
                    if trigger_count>1:
                        return None
                    self._seed_state += 1
            values = hps.values
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                collisions += 1
                if collisions <= self._max_collisions:
                    continue
                return None
            self._tried_so_far.add(values_hash)
            break
        print(trigger_count)
        return values

    def generate_hp_values_greedy(self,operation_list):#our greedy method
        best_hps = self._get_best_hps()

        collisions = 0
        while True:
            best_hp_name,best_hp_value_dict,special_sign=self._get_best_action(best_hps,operation_list,collisions,method='greedy')
            if special_sign !=None:
                write_opt(best_hp_name,best_hp_value_dict)
                return best_hps.values
            hps = kerastuner.HyperParameters()
            # Generate a set of random values.
            for hp in self.hyperparameters.space:
                hps.merge([hp])
                # if not active, do nothing.
                # if active, check if selected to be changed.
                if hps.is_active(hp):
                    # if was active and not selected, do nothing.
                    if best_hps.is_active(hp.name) and (hp.name not in best_hp_name):
                        hps.values[hp.name] = best_hps.values[hp.name]
                        continue
                    # if was not active or selected, sample.
                    hps.values[hp.name] = best_hp_value_dict[hp.name]#hp.random_sample(self._seed_state)
                    self._seed_state += 1
            values = hps.values
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                collisions += 1
                if collisions <= self._max_collisions:
                    continue
                return None
            self._tried_so_far.add(values_hash)
            break
        return values
    # zxy
    
    def _select_hps(self):#TODO::
        trie = Trie()
        best_hps = self._get_best_hps()
        for hp in best_hps.space:
            # Not picking the fixed hps for generating new values.
            if best_hps.is_active(hp) and not isinstance(
                hp, kerastuner.engine.hyperparameters.Fixed
            ):
                trie.insert(hp.name)
        all_nodes = trie.nodes

        if len(all_nodes) <= 1:
            return []

        probabilities = np.array([1 / node.num_leaves for node in all_nodes])
        sum_p = np.sum(probabilities)
        probabilities = probabilities / sum_p
        node = np.random.choice(all_nodes, p=probabilities)

        return trie.get_hp_names(node)

    def _next_initial_hps(self):
        for index, hps in enumerate(self.initial_hps):
            if not self._tried_initial_hps[index]:
                self._tried_initial_hps[index] = True
                return hps

    # def _populate_space(self, trial_id):
    #     if not all(self._tried_initial_hps):
    #         values = self._next_initial_hps()
    #         return {
    #             "status": kerastuner.engine.trial.TrialStatus.RUNNING,
    #             "values": values,
    #         }

    #     for i in range(self._max_collisions):
    #         hp_names = self._select_hps()
    #         values = self._generate_hp_values(hp_names)
    #         # Reached max collisions.
    #         if values is None:
    #             continue
    #         # Values found.
    #         return {
    #             "status": kerastuner.engine.trial.TrialStatus.RUNNING,
    #             "values": values,
    #         }
    #     # All stages reached max collisions.
    #     return {
    #         "status": kerastuner.engine.trial.TrialStatus.STOPPED,
    #         "values": None,
    #     }

    #zxy 
    def _populate_space(self, trial_id):
        if not all(self._tried_initial_hps):
            values = self._next_initial_hps()
            return {
                "status": kerastuner.engine.trial.TrialStatus.RUNNING,
                "values": values,
            }

        for i in range(self._max_collisions):
            values=self.obtain_new_hps()
            if values==None: # worst: use greedy
                print('============Fail to Generate!!!! USE GREEDY Method NOW!!!============')
                hp_names = self._select_hps()
                values = self._generate_hp_values(hp_names)
            # Reached max collisions.
            if values is None:
                continue
            # Values found.
            return {
                "status": kerastuner.engine.trial.TrialStatus.RUNNING,
                "values": values,
            }
        # All stages reached max collisions.
        return {
            "status": kerastuner.engine.trial.TrialStatus.STOPPED,
            "values": None,
        }
    # zxy

    def _get_best_hps(self):
        best_trials = self.get_best_trials()
        if best_trials:
            return best_trials[0].hyperparameters.copy()
        else:
            return self.hyperparameters.copy()

    def _generate_hp_values(self, hp_names): #TODO::
        best_hps = self._get_best_hps()

        collisions = 0
        while True:
            hps = kerastuner.HyperParameters()
            # Generate a set of random values.
            for hp in self.hyperparameters.space:
                hps.merge([hp])
                # if not active, do nothing.
                # if active, check if selected to be changed.
                if hps.is_active(hp):
                    # if was active and not selected, do nothing.
                    if best_hps.is_active(hp.name) and hp.name not in hp_names:
                        hps.values[hp.name] = best_hps.values[hp.name]
                        continue
                    # if was not active or selected, sample.
                    hps.values[hp.name] = hp.random_sample(self._seed_state)
                    self._seed_state += 1
            values = hps.values
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                collisions += 1
                if collisions <= self._max_collisions:
                    continue
                return None
            self._tried_so_far.add(values_hash)
            break
        return values


class Greedy(tuner_module.AutoTuner):
    def __init__(
        self,
        hypermodel: kerastuner.HyperModel,
        objective: str = "val_loss",
        max_trials: int = 10,
        initial_hps: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        hyperparameters: Optional[kerastuner.HyperParameters] = None,
        tune_new_entries: bool = True,
        allow_new_entries: bool = True,
        **kwargs
    ):
        self.seed = seed
        oracle = GreedyOracle(
            objective=objective,
            max_trials=max_trials,
            initial_hps=initial_hps,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)

# zxy 
# def judge_dirs(target_dir):
#     params_path=os.path.join(target_dir,'param.pkl')
#     gw_path=os.path.join(target_dir,'gradient_weight.pkl')
#     his_path=os.path.join(target_dir,'history.pkl')

#     with open(params_path, 'rb') as f:#input,bug type,params
#         hyperparameters = pickle.load(f)
#     with open(his_path, 'rb') as f:#input,bug type,params
#         history = pickle.load(f)
#     with open(gw_path, 'rb') as f:#input,bug type,params
#         gw = pickle.load(f)

#     arch=get_arch(hyperparameters)
#     loss=get_loss(history)
#     grad,wgt=get_gradient(gw)
    

#     return "{}-{}-{}-{}".format(arch,loss,grad,wgt)