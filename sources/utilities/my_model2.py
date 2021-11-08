import os
import json
from typing import List

import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers, models

# IMPORT THE MODEL API FROM WHICH YOUR MODEL MUST INHERITATE : 
try:
    from model_api import ModelApi
except:pass
try:
    from utilities.model_api import ModelApi
except:pass
try:
    from sources.utilities.model_api import ModelApi
except:pass
    
##################################################
## In this script, candidates are asked to implement two things:
#    1- the model, along with its training, prediction/inference methods;
#    2- the interface of this model with SKF evaluation and remote (AWS sagemaker) computation tools.
#
# See example notebook for an example of how to use this script
##################################################
## Author: François Caire
## Maintainer: François Caire
## Email: francois.caire at skf.com
##################################################

"""
BELOW IS THE IMPLEMENTATION OF YOUR MODEL

This example (GRUModel) implements a GRU class model inherited from a Pytorch model class.
Your solution can be any, the interface with higher level scripts needed (sagemaker_api and calc_metrics) 
as well as test_submission, is done in the class definition below (cf. MyModel def).

To conclude: the only constraint you have here is that your model is instanciated inside the MyModel class definition
where you have to implement all the methods required for the higher levels tools we need (and that you also need !) 
to evaluate your solution in terms of precision, inference and training times etc...
"""

class Keras_LSTM_Model():
    
    def __init__(self, gpu_available=False, input_shape=1, hidden_layer_dimension=40,steps_per_epoch=300,lr=0.01,
                 device='cpu', epochs=50, output_size=5, batch_size = 20):
        self.hidden_layer_dimension = hidden_layer_dimension
        self.output_size = output_size
        self.input_shape = input_shape
        self.epochs = epochs
        self.lr = lr
        self.steps_per_epoch=steps_per_epoch
        self.batch_size = batch_size
        # self.model = Sequential()
        # self.model.add(LSTM(self.hidden_layer_dimension, return_sequences=True, input_shape = (self.input_shape, 1)))
        # self.model.add(Dropout(0.2))
        # self.model.add(LSTM(self.hidden_layer_dimension, return_sequences=True))
        # self.model.add(Dropout(0.2))
        # self.model.add(LSTM(self.hidden_layer_dimension, return_sequences=True))
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(20))
        # self.model.add(Dense(10))
        # self.model.add(Dense(output_size, activation='softmax'))

        input_layer = Input(shape=(self.input_shape, 1))

        LSTM0 = LSTM(self.hidden_layer_dimension, return_sequences=True)(input_layer)
        drop_out1 = Dropout(0.2)(LSTM0)
        LSTM1 = LSTM(self.hidden_layer_dimension, return_sequences=True)(drop_out1)
        drop_out2 = Dropout(0.2)(LSTM1)
        LSTM2 = LSTM(self.hidden_layer_dimension, return_sequences=True)(drop_out2)
        drop_out3 = Dropout(0.2)(LSTM2)
        dense_up = Dense(20)(drop_out3)

        dense0 = Dense(10)(dense_up)
        output0 = Dense(1)(dense0)

        dense1 = Dense(10)(dense_up)
        output1 = Dense(1)(dense1)

        dense2 = Dense(10)(dense_up)
        output2 = Dense(1)(dense2)

        dense3 = Dense(10)(dense_up)
        output3 = Dense(1)(dense3)

        dense4 = Dense(10)(dense_up)
        output4 = Dense(1)(dense4)

        self.model = Model(inputs = input_layer, outputs = [output0, output1, output2, output3, output4])
        
        opt = optimizers.RMSprop(learning_rate=self.lr)
        self.model.compile(optimizer=opt, loss='mse')
        print(self.model.summary())


    def fit(self, X_train, y_train):
        X_train = X_train[0].reshape(X_train[0].shape[0], 1, 1)
        return self.model.fit(X_train,
                            y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            steps_per_epoch = self.steps_per_epoch,
                            verbose=1)

    
    def predict(self, X_test):
        # y_pred = self.model.predict(X_test)
        # y_pred = np.concatenate(y_pred, axis=1)
        # y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        # print("y_pred type: "+str(type(y_pred)))
        # return y_pred
        return self.model.predict(X_test)

"""
BELOW IS THE IMPLEMENTATION OF YOUR MODEL'S INTERFACE

Here you have to implement all the necessary methods as they are 
defined in the parent class ModelApi (cf file model_api.py).
These methods are used in higher levels scripts such as:
    - sagemaker_api.py that allows you (and us) to run training tasks on local or Amazon specific instances ;
    - calc_metrics.py/calc_metrics_on_sagemaker.py that allows you (and us) to compute the performance metrics of your solution, given your model definition (this file) and a test dataset as input;
"""
class MyModel(ModelApi):

    def __init__(self, **model_kwargs):
        self.model_kwargs = model_kwargs
        
        self.lstm_model: Keras_LSTM_Model = Keras_LSTM_Model(**model_kwargs)

    def fit(self, xs: List[np.ndarray], ys: List[np.ndarray], timeout=36000):
        return self.lstm_model.fit(xs, ys)

    @classmethod
    def get_sagemaker_estimator_class(self):
        """
        return the class with which to initiate an instance on sagemaker:
        e.g. SKLearn, PyTorch, TensorFlow, etc.
        by default - use SKLearn image.
        """
        
        from sagemaker.tensorflow import TensorFlow
        framework_version = '2.3.0'
        
        return TensorFlow,framework_version

    def predict_one_timepoint(self, x: float) -> np.ndarray:
        y_pred = self.lstm_model.predict(np.asarray([x]))
        y_pred = np.concatenate(y_pred, axis=1)
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        return y_pred

    def predict_timeseries(self, x: np.ndarray) -> np.ndarray:
        y_pred = self.lstm_model.predict(x)
        y_pred = np.concatenate(y_pred, axis=1)
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        return y_pred

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, 'model_kwargs.json')
        with open(path, 'w') as f:
            json.dump(self.model_kwargs, f)

        # recommended way from http://pytorch.org/docs/master/notes/serialization.html
        self.lstm_model.model.save(model_dir)

    @classmethod
    def load(cls, model_dir: str):
        path = os.path.join(model_dir, 'model_kwargs.json')
        with open(path, 'r') as f:
            model_kwargs = json.load(f)
            
        # model_kwargs['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        my_model = cls(**model_kwargs)
        # with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        my_model.lstm_model = models.load_model(model_dir)
        # my_model = models.load_model(model_dir)

        return my_model

    @classmethod
    def create_model(cls, gpu_available: bool = False, **kwargs):
        return cls(**kwargs)

    @property
    def description(self):
        team_name = 'Guan_Li_DSIA_5012B'  #NO SPACE ALLOWED
        email = 'xianli.li@edu.esiee.fr'
        model_name='Keras_model'
        model_title = 'Keras Model LSTM'
        affiliation = 'ESIEE Paris'
        description = 'LSTM'
        technology_stack = 'Keras'
        other_remarks = ""


        return dict(team_name=team_name,
                    email=email,
                    model_name=model_name,
                    model_title=model_title,
                    description=description,
                    technology_stack=technology_stack,
                    other_remarks=other_remarks,
                    affiliation=affiliation)
