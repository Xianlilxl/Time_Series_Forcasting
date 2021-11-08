try:
    from model_api import ModelApi
except:pass
try:
    from utilities.model_api import ModelApi
except: pass
try:
    from sources.utilities.model_api import ModelApi
except:pass

from typing import List
# from keras import models
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers, models
import os
import json

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
        # self.model.add(Dropout(0.3))
        # self.model.add(LSTM(self.hidden_layer_dimension, return_sequences=True))
        # self.model.add(Dropout(0.3))
        # self.model.add(Dense(output_size))

        input_layer = Input(shape=(self.input_shape, 1))

        LSTM1 = LSTM(self.hidden_layer_dimension, return_sequences=True)(input_layer)
        drop_out1 = Dropout(0.2)(LSTM1)
        LSTM2 = LSTM(self.hidden_layer_dimension, return_sequences=True)(drop_out1)
        drop_out2 = Dropout(0.2)(LSTM2)
        LSTM3 = LSTM(self.hidden_layer_dimension, return_sequences=True)(drop_out2)
        drop_out3 = Dropout(0.2)(LSTM3)
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


    def fit(self, X_train, y_train, X_valid=[], y_valid=[]):
        X_train = X_train[0].reshape(X_train[0].shape[0], 1, 1)
        if X_valid ==[]:
            return self.model.fit(X_train,
                    y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    steps_per_epoch = self.steps_per_epoch,
                    verbose=1)
        else:
            earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
            return self.model.fit(X_train,
                        y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        steps_per_epoch = self.steps_per_epoch,
                        callbacks=[earlystop],
                        validation_data=(X_valid, y_valid),
                        verbose=1)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
        

class MyModel(ModelApi):
    """Interface for Magnetic Bearing Model. All the following methods,
    must be implemented by your model
    """
    def __init__(self, **model_kwargs):
        self.model_kwargs = model_kwargs
        
        self.lstm_model: Keras_LSTM_Model = Keras_LSTM_Model(**model_kwargs)

    @classmethod
    def create_model(cls, **kwargs):
        """

        :param gpu_available: whether a gpu device is available.
        :return: an instance of your model. Use "return cls(*your_args, **your_kwargs)" to call your constructor.
        Note: create_model is expected to take no more than 30 seconds
        """

        return cls(**kwargs)

    @property
    def description(self):
        """
        :return: a dictionary with the following properties
        """

        team_name = 'Guan_Li_DSIA_5012B'  #NO SPACE ALLOWED
        email = 'xianli.li@edu.esiee.fr'
        model_title = 'Keras Model LSTM'
        affiliation = 'ESIEE Paris'
        description = 'LSTM'
        technology_stack = 'Keras'
        other_remarks = ""

        return dict(team_name=team_name,
                    email=email,
                    model_title=model_title,
                    description=description,
                    technology_stack=technology_stack,
                    other_remarks=other_remarks,
                    affiliation=affiliation)

    @classmethod
    def get_sagemaker_estimator_class(self):
        """
        return the class with which to initiate an instance on sagemaker:
        e.g. SKLearn, PyTorch, TensorFlow, etc.
        by default - use SKLearn image.

        """

        # Implementation examplea:
        """
        from sagemaker.sklearn import SKLearn
        return SKLearn
        """

        # or

        """
        from sagemaker.pytorch import PyTorch
        return PyTorch
        """

        from sagemaker.tensorflow import TensorFlow
        framework_version = '2.3.0'
        
        return TensorFlow,framework_version

    def fit(self,  xs: List[np.ndarray], ys: List[np.ndarray],  timeout=36000):
        """ train on several (x, y) examples
        :param xs: input data given as a list of series. Each series is a 2D ndarray with rows representing samples, and columns represening features.
        :param ys: labels (could be multi-label) for each given series. Each labels series is a 2D ndarray with rows representing samples, and columns represening output labels.
        :param timeout: maximal time (on the hosting machine) allowed for this operation (in seconds).
        """
        return self.lstm_model.fit(xs, ys)

    def predict_one_timepoint(self, x: float) -> np.ndarray:
        """ produce a prediction: x -> y where x is 1 sample
        :param x: input vector 1D ndarray with shape (1,-1)
        :return:  corresponding output vector as 2D np.ndarray 2D ndarray with shape (1,-1)

        Note: calling predict_one_timepoint may change model's state (e.g. to save history).
        Note: predict_one_timepoint may assume that predict_one_timepoint was called sequentially on all previous timepoints
        Note: predict_one_timepoint is expected to take no more than 1 second
        """

        return self.lstm_model.predict(x)

    def predict_timeseries(self, x: np.ndarray) -> np.ndarray:
        """ produce a prediction: x -> y where x is the entire time series from the beginning

        :param x: 1 input series given as a 2D ndarray with rows representing samples, and columns representing features.
        :return:  corresponding predictions as 2D np.ndarray

        Note: calling predict_series may change model's state.
        Note: self.predict_series(x) should return the same results as [self.predict_one_timepoint(xi) for xi in x] up to 5 digits precision.
        Note: predict_timeseries is expected to take no more than 1 second per sample
        """

        return self.lstm_model.predict(x)


    def save(self, model_dir:str):
        """ save the model to a file
        :param path: a path to the file which will store the model

        Note: save is expected to take no more than 10 minutes
        """

        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, 'model_kwargs.json')
        with open(path, 'w') as f:
            json.dump(self.model_kwargs, f)

        # recommended way from http://pytorch.org/docs/master/notes/serialization.html
        self.lstm_model.model.save(model_dir)

    @classmethod
    def load(cls, model_dir:str):#->ModelApi
        """ load a pretrained model from a file
        :param path: path to the file of the model.
        :return: an instance of this class initialized with loaded model.

        Note: save is expected to take no more than 10 minutes
        """

        path = os.path.join(model_dir, 'model_kwargs.json')
        with open(path, 'r') as f:
            model_kwargs = json.load(f)
            
        # model_kwargs['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        my_model = cls(**model_kwargs)
        # with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        my_model.lstm_model = models.load_model(model_dir)
        # my_model = models.load_model(model_dir)

        return my_model
