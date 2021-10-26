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
import numpy as np
from keras.models import Model, Sequential
from keras.layers import GRU, Dense, LSTM, Dropout, Input
from keras.callbacks import EarlyStopping
import torch 


class Keras_GRU_Model():
    
    def __init__(self, input_shape=1, hidden_layer_dimension=[20, 40],
                 epochs=50, output_size=5, batch_size = 5):
        self.hidden_layer_dimension = hidden_layer_dimension
        self.output_size = output_size
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        # self.model.add(LSTM(self.hidden_layer_dimension[0], return_sequences=True, input_shape = (self.input_shape, 1)))
        # self.model.add(Dropout(0.3))
        # self.model.add(LSTM(self.hidden_layer_dimension[1], return_sequences=True))
        # self.model.add(Dropout(0.3))
        # self.model.add(Dense(output_size))

        input_layer = Input(shape=(self.input_shape, 1))

        LSTM0 = LSTM(self.hidden_layer_dimension[0], return_sequences=True)(input_layer)
        output0 = Dense(1)(LSTM0)

        LSTM1 = LSTM(self.hidden_layer_dimension[0], return_sequences=True)(input_layer)
        output1 = Dense(1)(LSTM1)

        LSTM2 = LSTM(self.hidden_layer_dimension[0], return_sequences=True)(input_layer)
        output2 = Dense(1)(LSTM2)

        LSTM3 = LSTM(self.hidden_layer_dimension[0], return_sequences=True)(input_layer)
        output3 = Dense(1)(LSTM3)

        LSTM4 = LSTM(self.hidden_layer_dimension[0], return_sequences=True)(input_layer)
        output4 = Dense(1)(LSTM4)

        self.model = Model(inputs = input_layer, outputs = [output0, output1, output2, output3, output4])
        
        self.model.compile(optimizer='RMSprop', loss='mse')
        print(self.model.summary())


    def fit(self, X_train, y_train, X_valid, y_valid):
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
        return self.model.fit(X_train,
                    y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    callbacks=[earlystop],
                    validation_data=(X_valid, y_valid),
                    verbose=1)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
        

class Keras_Model(ModelApi):
    """Interface for Magnetic Bearing Model. All the following methods,
    must be implemented by your model
    """
    def __init__(self, **model_kwargs):
        self.model_kwargs = model_kwargs
        
        self.gru_model: Keras_GRU_Model = Keras_GRU_Model(**model_kwargs)

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

        team_name = 'team_name'  #NO SPACE ALLOWED
        email = 'your_email@gmail.com'
        model_title = 'Model title - e.g. My Favorite Model'
        affiliation = 'Company/Instituition'
        description = 'description of the model and architecture'
        technology_stack = 'technology stack you are using, e.g. sklearn, pytorch'
        other_remarks = "put here anything else you'd like us to know"

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

        raise NotImplementedError('model get_sagemaker_estimator_class() was not implemented')

    def fit(self,  X_train: List[np.ndarray], y_train: List[np.ndarray], X_valid: List[np.ndarray], y_valid: List[np.ndarray], timeout=36000):
        """ train on several (x, y) examples
        :param xs: input data given as a list of series. Each series is a 2D ndarray with rows representing samples, and columns represening features.
        :param ys: labels (could be multi-label) for each given series. Each labels series is a 2D ndarray with rows representing samples, and columns represening output labels.
        :param timeout: maximal time (on the hosting machine) allowed for this operation (in seconds).
        """
        return self.gru_model.fit(X_train, y_train, X_valid, y_valid)

    def predict_one_timepoint(self, x: float) -> np.ndarray:
        """ produce a prediction: x -> y where x is 1 sample
        :param x: input vector 1D ndarray with shape (1,-1)
        :return:  corresponding output vector as 2D np.ndarray 2D ndarray with shape (1,-1)

        Note: calling predict_one_timepoint may change model's state (e.g. to save history).
        Note: predict_one_timepoint may assume that predict_one_timepoint was called sequentially on all previous timepoints
        Note: predict_one_timepoint is expected to take no more than 1 second
        """

        return self.gru_model.predict(x)

    def predict_timeseries(self, x: np.ndarray) -> np.ndarray:
        """ produce a prediction: x -> y where x is the entire time series from the beginning

        :param x: 1 input series given as a 2D ndarray with rows representing samples, and columns representing features.
        :return:  corresponding predictions as 2D np.ndarray

        Note: calling predict_series may change model's state.
        Note: self.predict_series(x) should return the same results as [self.predict_one_timepoint(xi) for xi in x] up to 5 digits precision.
        Note: predict_timeseries is expected to take no more than 1 second per sample
        """

        return self.gru_model.predict(x)


    def save(self, model_dir:str):
        """ save the model to a file
        :param path: a path to the file which will store the model

        Note: save is expected to take no more than 10 minutes
        """

        raise NotImplementedError('model predict() was not implemented')

    @classmethod
    def load(cls, model_dir:str):#->ModelApi
        """ load a pretrained model from a file
        :param path: path to the file of the model.
        :return: an instance of this class initialized with loaded model.

        Note: save is expected to take no more than 10 minutes
        """

        raise NotImplementedError('model load() was not implemented')