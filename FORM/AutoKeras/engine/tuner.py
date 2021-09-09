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

import copy
import os

import kerastuner
import tensorflow as tf
from kerastuner.engine import hypermodel as hm_module
from tensorflow.keras import callbacks as tf_callbacks
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.util import nest

from autokeras import pipeline as pipeline_module
from autokeras.utils import data_utils
from autokeras.utils import utils

import tensorflow.keras as keras
import pickle
import numpy as np

class LossHistory(keras.callbacks.Callback):

    def __init__(self,training_data,model,total_epoch,batch_size,save_path): #only support epoch method now
        """[summary]

        Args:
            training_data ([list]): [training dataset]
            model ([model]): [untrained model]
            batch_size ([int]): [batch size]
            save-dir([str]):[the dir to save the detect result]
            checktype (str, optional): [checktype,'a_b', a can be chosen from ['epoch', 'batch'], b is number, it means the monitor will check \
            the gradient and loss every 'b' 'a'.]. Defaults to 'epoch_5'.
            satisfied_acc (float, optional): [the satisfied accuracy, when val accuracy beyond this, the count ++, when the count is bigger or\
                equal to satisfied_count, training stop.]. Defaults to 0.7.

        """
        self.trainX,self.trainy,self.testX,self.testy = read_data(training_data,batch_size)
        self.model=model
        self.epoch=total_epoch
        self.save_path=save_path
        save_dict={}
        save_dict['gradient']={}
        save_dict['weight']={}
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)

        self.x_path='./Test_dir/tmp/x.npy'
        self.y_path='./Test_dir/tmp/y.npy'
        self.model_path='./Test_dir/tmp/model.h5'
        trainingExample = self.trainX
        trainingY=self.trainy
        np.save('./Test_dir/tmp/x.npy',trainingExample)
        np.save('./Test_dir/tmp/y.npy',trainingY)


    def on_epoch_end(self,epoch,logs={}):
        self.model.save('./Test_dir/tmp/model.h5')
        get_gradient(self.model_path,self.x_path,self.y_path,epoch,self.save_path)

    def on_train_end(self,logs=None):
        print('Finished Training')


def get_gradient(model_path,x_path,y_path,epoch,save_path):
    import subprocess
    command=".your_python_interpreter/python ./utils/get_gradient_on_cpu.py -m {} -dx {} -dy {} -ep {} -sp {}" #TODO:need to set your your python interpreter path

    out_path=save_path.split('.')[0]+'_out'
    out_file = open(out_path, 'w')
    out_file.write('logs\n')
    run_cmd=command.format(model_path,x_path,y_path,epoch,save_path)
    subprocess.Popen(run_cmd, shell=True, stdout=out_file, stderr=out_file)

class AutoTuner(kerastuner.engine.tuner.Tuner):
    """A Tuner class based on KerasTuner for AutoKeras.

    Different from KerasTuner's Tuner class. AutoTuner's not only tunes the
    Hypermodel which can be directly built into a Keras model, but also the
    preprocessors. Therefore, a HyperGraph stores the overall search space containing
    both the Preprocessors and Hypermodel. For every trial, the HyperGraph build the
    PreprocessGraph and KerasGraph with the provided HyperParameters.

    The AutoTuner uses EarlyStopping for acceleration during the search and fully
    train the model with full epochs and with both training and validation data.
    The fully trained model is the best model to be used by AutoModel.

    # Arguments
        oracle: kerastuner Oracle.
        hypermodel: kerastuner KerasHyperModel.
        **kwargs: The args supported by KerasTuner.
    """

    def __init__(self, oracle, hypermodel, **kwargs):
        # Initialize before super() for reload to work.
        self._finished = False
        super().__init__(oracle, hypermodel, **kwargs)
        # Save or load the HyperModel.
        self.hypermodel.hypermodel.save(os.path.join(self.project_dir, "graph"))
        self.hyper_pipeline = None

    def _populate_initial_space(self):
        # Override the function to prevent building the model during initialization.
        return

    def get_best_model(self):
        with hm_module.maybe_distribute(self.distribution_strategy):
            model = tf.keras.models.load_model(self.best_model_path)
        return model

    def get_best_pipeline(self):
        return pipeline_module.load_pipeline(self.best_pipeline_path)

    def _pipeline_path(self, trial_id):
        return os.path.join(self.get_trial_dir(trial_id), "pipeline")

    def _prepare_model_build(self, hp, **kwargs):
        """Prepare for building the Keras model.

        It build the Pipeline from HyperPipeline, transform the dataset to set
        the input shapes and output shapes of the HyperModel.
        """
        dataset = kwargs["x"]
        pipeline = self.hyper_pipeline.build(hp, dataset)
        pipeline.fit(dataset)
        dataset = pipeline.transform(dataset)
        self.hypermodel.hypermodel.set_io_shapes(data_utils.dataset_shape(dataset))

        if "validation_data" in kwargs:
            validation_data = pipeline.transform(kwargs["validation_data"])
        else:
            validation_data = None
        return pipeline, dataset, validation_data

    def _build_and_fit_model(self, trial, fit_args, fit_kwargs):
        (
            pipeline,
            fit_kwargs["x"],
            fit_kwargs["validation_data"],
        ) = self._prepare_model_build(trial.hyperparameters, **fit_kwargs)
        pipeline.save(self._pipeline_path(trial.trial_id))


        import uuid
        import os
        import time
        import pickle
        import sys
        import shutil
        sys.path.append('./utils')
        from load_test_utils import traversalDir_FirstDir,read_opt,write_algw
        root_path='./Test_dir/demo_result'
        root_path=os.path.abspath(root_path)
        # get greedy ak path:
        origin_path=os.path.join(root_path,'origin')
        log_path=os.path.join(root_path,'log.pkl')
        with open(log_path, 'rb') as f:#input,bug type,params
            log_dict = pickle.load(f)
        params_path=log_dict['param_path']
             
        if traversalDir_FirstDir(root_path)==[]:
            with open(params_path, 'rb') as f:#input,bug type,params
                ak_params = pickle.load(f)
            trial.hyperparameters=ak_params
            self.oracle.hyperparameters=ak_params

        
        model = self.hypermodel.build(trial.hyperparameters)
        self.adapt(model, fit_kwargs["x"])


        read_path=os.path.abspath('./Test_dir/tmp/tmp_action_value.pkl')
        if os.path.exists(read_path):
            special=True
            model=read_opt(model)# check for special actions like activation and initializer
        else:
            special=False
        

        msg={}
        msg['epochs']=fit_kwargs['epochs']
        data,batch_size=extract_dataset(list(fit_kwargs['x'].as_numpy_iterator()),fit_kwargs['validation_data'].as_numpy_iterator(),method='cifar')
        # msg['data']=data
        msg['batch']=batch_size

        save_path=os.path.abspath('./Test_dir/tmp/gradient_weight.pkl')
        predictor=LossHistory(training_data=data,model=model,total_epoch=fit_kwargs['epochs'],batch_size=batch_size,save_path=save_path)
        fit_kwargs['callbacks'].append(predictor)

        

        _, history = utils.fit_with_adaptive_batch_size(
            model, self.hypermodel.hypermodel.batch_size, **fit_kwargs
        )

        train_history=history.history
        max_val_acc=max(train_history['val_accuracy'])
        trial_num_path=os.path.join(os.path.dirname(root_path),'num.pkl')
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        if not os.path.exists(log_path):
            log_dict={}
            log_dict['cur_trial']=0
            
        else:
            with open(log_path, 'rb') as f:#input,bug type,params
                log_dict = pickle.load(f)
                log_dict['cur_trial']+=1
                current_time=time.time()-log_dict['start_time']

        new_dir_name=str(log_dict['cur_trial'])+'-'+str(round(max_val_acc,2))+str(uuid.uuid3(uuid.NAMESPACE_DNS,str(time.time())))[-13:]

        new_dir_path=os.path.join(root_path,new_dir_name)
        
        log_dict[new_dir_name]={}
        log_dict[new_dir_name]['time']=current_time

        with open(log_path, 'wb') as f:
            pickle.dump(log_dict, f)

        trial_id_path=os.path.join(root_path,'trial_id.pkl')
        if os.path.exists(trial_id_path):
            with open(trial_id_path, 'rb') as f:#input,bug type,params
                trial_id_dict = pickle.load(f)
        else:
            trial_id_dict={}
        trial_id_dict[trial.trial_id]=new_dir_path
        with open(trial_id_path, 'wb') as f:
            pickle.dump(trial_id_dict, f)   


        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)
        else:
            new_dir_name=str(log_dict['cur_trial'])+'-'+str(round(max_val_acc,2))+str(uuid.uuid3(uuid.NAMESPACE_DNS,str(time.time())))[-13:]
            new_dir_path=os.path.join(root_path,new_dir_name)
            os.makedirs(new_dir_path)
        
        model_path=os.path.join(new_dir_path,'model.h5')
        history_path=os.path.join(new_dir_path,'history.pkl')
        hyperparam_path=os.path.join(new_dir_path,'param.pkl')
        message_path=os.path.join(new_dir_path,'msg.pkl')
        hm_path = os.path.join(new_dir_path,'hypermodel.pkl')
        new_save_path=os.path.join(new_dir_path,'gradient_weight.pkl')

        if special:
            new_read_path=os.path.join(new_dir_path,'tmp_action_value.pkl')
            
            shutil.move(read_path,new_read_path)

        shutil.move(save_path,new_save_path)

        with open(hm_path, 'wb') as f:
            pickle.dump(self.hypermodel, f)   
        
        with open(message_path, 'wb') as f:
            pickle.dump(msg, f)

        try:
            model.save(model_path)
        except Exception as e:
            print(e)
            model.save(new_dir_path, save_format='tf')

        with open(history_path, 'wb') as f:
            pickle.dump(train_history, f)
        with open(hyperparam_path, 'wb') as f:
            pickle.dump(trial.hyperparameters, f)
        
        # print(1)
        # if current_time>=7200 and log_dict['cur_trial']>=20:
        if current_time>=14000 :#and log_dict['cur_trial']>=14
            with open(trial_num_path, 'rb') as f:#input,bug type,params
                num = pickle.load(f)
            num+=1
            with open(trial_num_path, 'wb') as f:
                pickle.dump(num, f)
            
            new_dir='{}_{}'.format(root_path,num)
            os.rename(root_path,new_dir)
            print('finish')
            os._exit(0)

        write_algw(new_dir_path)

        return history

    @staticmethod
    def adapt(model, dataset):
        """Adapt the preprocessing layers in the model."""
        # Currently, only support using the original dataset to adapt all the
        # preprocessing layers before the first non-preprocessing layer.
        # TODO: Use PreprocessingStage for preprocessing layers adapt.
        # TODO: Use Keras Tuner for preprocessing layers adapt.
        x = dataset.map(lambda x, y: x)

        def get_output_layer(tensor):
            tensor = nest.flatten(tensor)[0]
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                if not isinstance(layer, preprocessing.PreprocessingLayer):
                    break
                input_node = nest.flatten(layer.input)[0]
                if input_node is tensor:
                    return layer
            return None

        for index, input_node in enumerate(nest.flatten(model.input)):
            temp_x = x.map(lambda *args: nest.flatten(args)[index])
            layer = get_output_layer(input_node)
            while layer is not None:
                if isinstance(layer, preprocessing.PreprocessingLayer):
                    layer.adapt(temp_x)
                temp_x = temp_x.map(layer)
                layer = get_output_layer(layer.output)
        return model

    def search(self, epochs=None, callbacks=None, validation_split=0, **fit_kwargs):
        """Search for the best HyperParameters.

        If there is not early-stopping in the callbacks, the early-stopping callback
        is injected to accelerate the search process. At the end of the search, the
        best model will be fully trained with the specified number of epochs.

        # Arguments
            callbacks: A list of callback functions. Defaults to None.
            validation_split: Float.
        """
        if self._finished:
            return

        if callbacks is None:
            callbacks = []

        self.hypermodel.hypermodel.set_fit_args(validation_split, epochs=epochs)

        # Insert early-stopping for adaptive number of epochs.
        epochs_provided = True
        if epochs is None:
            epochs_provided = False
            epochs = 1000
            if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
                callbacks.append(
                    tf_callbacks.EarlyStopping(patience=10, min_delta=1e-4)
                )

        # Insert early-stopping for acceleration.
        early_stopping_inserted = False
        new_callbacks = self._deepcopy_callbacks(callbacks)
        if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
            early_stopping_inserted = True
            new_callbacks.append(
                tf_callbacks.EarlyStopping(patience=10, min_delta=1e-4)
            )

        # Populate initial search space.
        hp = self.oracle.get_space()
        self._prepare_model_build(hp, **fit_kwargs)
        self.hypermodel.build(hp)
        self.oracle.update_space(hp)

        super().search(epochs=epochs, callbacks=new_callbacks, **fit_kwargs)

        # Train the best model use validation data.
        # Train the best model with enough number of epochs.
        if validation_split > 0 or early_stopping_inserted:
            copied_fit_kwargs = copy.copy(fit_kwargs)

            # Remove early-stopping since no validation data.
            # Remove early-stopping since it is inserted.
            copied_fit_kwargs["callbacks"] = self._remove_early_stopping(callbacks)

            # Decide the number of epochs.
            copied_fit_kwargs["epochs"] = epochs
            if not epochs_provided:
                copied_fit_kwargs["epochs"] = self._get_best_trial_epochs()

            # Concatenate training and validation data.
            if validation_split > 0:
                copied_fit_kwargs["x"] = copied_fit_kwargs["x"].concatenate(
                    fit_kwargs["validation_data"]
                )
                copied_fit_kwargs.pop("validation_data")

            self.hypermodel.hypermodel.set_fit_args(
                0, epochs=copied_fit_kwargs["epochs"]
            )
            pipeline, model = self.final_fit(**copied_fit_kwargs)
        else:
            model = self.get_best_models()[0]
            pipeline = pipeline_module.load_pipeline(
                self._pipeline_path(self.oracle.get_best_trials(1)[0].trial_id)
            )

        model.save(self.best_model_path)
        pipeline.save(self.best_pipeline_path)
        self._finished = True

    def get_state(self):
        state = super().get_state()
        state.update({"finished": self._finished})
        return state

    def set_state(self, state):
        super().set_state(state)
        self._finished = state.get("finished")

    @staticmethod
    def _remove_early_stopping(callbacks):
        return [
            copy.deepcopy(callbacks)
            for callback in callbacks
            if not isinstance(callback, tf_callbacks.EarlyStopping)
        ]

    def _get_best_trial_epochs(self):
        best_trial = self.oracle.get_best_trials(1)[0]
        # steps counts from 0, so epochs = step + 1.
        return self.oracle.get_trial(best_trial.trial_id).best_step + 1

    def _build_best_model(self):
        best_trial = self.oracle.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters
        return self.hypermodel.build(best_hp)

    def final_fit(self, **kwargs):
        best_trial = self.oracle.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters
        pipeline, kwargs["x"], kwargs["validation_data"] = self._prepare_model_build(
            best_hp, **kwargs
        )

        model = self._build_best_model()
        self.adapt(model, kwargs["x"])
        model, _ = utils.fit_with_adaptive_batch_size(
            model, self.hypermodel.hypermodel.batch_size, **kwargs
        )
        return pipeline, model

    @property
    def best_model_path(self):
        return os.path.join(self.project_dir, "best_model")

    @property
    def best_pipeline_path(self):
        return os.path.join(self.project_dir, "best_pipeline")

    @property
    def objective(self):
        return self.oracle.objective

    @property
    def max_trials(self):
        return self.oracle.max_trials



def read_data(dataset,batch_size):
    # read data from a new unzipped dataset.
    trainX=dataset['x'][:batch_size]
    trainy=dataset['y'][:batch_size]
    testX=dataset['x_val'][:batch_size]
    testy=dataset['y_val'][:batch_size]
    return trainX,trainy,testX,testy

def extract_dataset(data_x,data_val,method='mnist'):
    tmp_path=os.path.abspath('./Test_dir/tmp/{}.pkl'.format(method))
    if os.path.exists(tmp_path):
        with open(tmp_path, 'rb') as f:#input,bug type,params
            dataset = pickle.load(f)
        batch_size=dataset['batch']
        del dataset['batch']
    else:
        dataset={}
        dataset['x']=[]
        dataset['y']=[]
        dataset['x_val']=[]
        dataset['y_val']=[]
        batch_size=data_x[0][0].shape[0]
        # if data_format==True:
        i = data_x[0]
        try:
            _=i[0].shape[1]
            tmp_i=i[0]
        except:
            tmp_i=i[0].reshape((-1,1))
        
        if dataset['x']==[]:
            dataset['x']=tmp_i
        if dataset['y']==[]:
            dataset['y']=i[1]          
        dataset['batch']=batch_size
        with open(tmp_path, 'wb') as f:
            pickle.dump(dataset, f)
        del dataset['batch']
    return dataset,batch_size