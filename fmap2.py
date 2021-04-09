#!/usr/bin/env python
import sys,os
import numpy as np
import pylab as py
import warnings
warnings.filterwarnings("ignore")
from pathlib import PurePath

#--local
from tools import checkdir,save,load

#--sklearn
from sklearn.preprocessing import StandardScaler,MinMaxScaler

#--tensorflow
import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers        as kl
import tensorflow.keras.models        as km
import tensorflow.keras.optimizers    as ko
import tensorflow.keras.callbacks     as kc
import tensorflow.keras.regularizers  as kr
import tensorflow.keras.backend as kb

import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"

"""
The forward mapper for CNF project.

This architecture is focused on example 2 (DIS sample).
"""

class CheckpointEveryK(tf.keras.callbacks.Callback):
    """
    Saves the model at certain checkpoints.
    """
    def __init__(self, period, model_n):
        super().__init__()
        self._period = period
        self._model_n = PurePath(model_n)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._period == 0:
            current_path = self._model_n / 'epoch_{}'.format(epoch)
            os.makedirs(current_path)
            self.model.save(str(current_path))

    def on_train_end(self, logs=None):
        """ Always save the final model. """
        current_path = self._model_n / 'final'
        os.makedirs(current_path)
        self.model.save(str(current_path))

class FMAP(object):
    """
    Architecture of the neural network (forward mapper).
    """
    def __init__(self,model,outdir='.'):
        """
        Initializes the model.
        """
        checkdir(outdir)
        self.model_name=model
        self.outdir=outdir

    def load_data(self,X,Y):
        """
        Loads the given data set.
        """
        self.X = X
        self.Y = Y
        self.normalize_data()

    def normalize_data(self):
        """
        Normalizes the training parameters and cross sections.
        """
        self.ssx = StandardScaler()
        self.X0 = self.ssx.fit_transform(self.X)

        self.ssy = MinMaxScaler()
        self.Y0 = self.ssy.fit_transform(self.Y)

    def set_ml1(self):
        """
        The architecture of the first training model.
        """
        #--network parameters
        original_dim= self.Y0.shape[1]
        input_shape = (original_dim, )
        # update output_dim if data has a different number of cross sections
        output_dim = 1329 #number of cross sections in training set

        #--model
        inputs = kl.Input(shape=input_shape, name='Y0')
        hidden = kl.Dense(1000, activation='relu')(inputs)
        hidden = kl.Dense(1000, activation='relu')(hidden)
        output = kl.Dense(output_dim, name = 'X0')(hidden)

        self.model=km.Model(inputs, output)

        self.loss = 'mean_squared_error'
        self.input  = self.Y0
        self.output = self.X0

    def set_ml2(self):
        """
        The architecture of the second training model.

        This is used so any changes can be trained and tested simulataneously
        to the original model.
        """
        #--network parameters
        original_dim= self.Y0.shape[1]
        inter_dim =1000
        input_shape = (original_dim, )
        # update output_dim if data has a different number of cross sections
        output_dim=1329 #number of cross sections in training set

        #--model
        inputs = kl.Input(shape=input_shape, name='Y0')
        x = kl.Dense(500, activation='relu')(inputs)
        x = kl.Dense(800,activation='relu')(x)
        x = kl.Dense(inter_dim,activation='relu')(x)
        output = kl.Dense(output_dim, name='X0')(x)

        self.model=km.Model(inputs,output, name='ml')

        self.loss= 'mean_squared_error'
        self.input  = self.Y0
        self.output = self.X0

    def set_model(self):
        if self.model_name.split("-")[0]=="ML1"   : self.set_ml1()
        elif self.model_name.split("-")[0]=="ML2"   : self.set_ml2()
        else:
            msg="Model = %s not supported"%self.model_name
            sys.exit(msg)

    def print_model(self):
        """
        Prints summary of model.
        """
        self.model.summary()

    def compile_model(self):
        """
        Compiles the model according to specific hyperparameters.
        """
        model_optimizer=ko.Adam(
          lr=1e-4, #learing rate
          beta_1=0.9,
          beta_2=0.999,
          epsilon=1e-08,
          decay=0.00001,
          )

        self.model.compile(
          optimizer=model_optimizer,
          loss=self.loss,
          metrics=['acc'],
          )

    def train(self,epochs=3):
        """
        Starts the training of the given model with a specific epoch number.
        """
        self.set_model()
        self.compile_model()
        self.print_model()

        mon = kc.EarlyStopping(
                monitor='val_loss',
                min_delta=0.0001,
                patience=600
                )

        history = self.model.fit(
        self.input,
        self.output,
        epochs=epochs,
        batch_size=256,
        validation_split=0.20,
        callbacks=[mon,CheckpointEveryK(100, self.model_name)])

        self.model.save('%s/%s-%d.h5'%(self.outdir,self.model_name,epochs))
        np.save('%s/%s-%d'%(self.outdir,self.model_name,epochs), history.history)
        return ('%s-%d'%(self.model_name,epochs))

    def load_model(self):
        """
        Loads trained model.
        """
        #load model to get all the self defined there
        self.set_model()
        if  self.model_name[:2]=='ML':
            custom_objects =None

        self.model= km.load_model(
            '%s/%s.h5'%(self.outdir,self.model_name),
            custom_objects=custom_objects
            )
