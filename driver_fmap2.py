#!/usr/bin/env python
import sys,os
import numpy as np
from numpy.random import default_rng
import pandas as pd
import seaborn as sns
#--matplolib
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
matplotlib.rc('text', usetex = True)
import pylab as py

from scipy.stats import pearsonr
from tools import load,save,checkdir
from scipy.stats import sem
from scipy import stats
"""
Driver for the data generating forward mapper for CNF project.
"""

def train_fmap(model,epochs):
    """
    Trains a model for a given number of epochs.

    params-
        model - name of model
        epochs - number of epochs for training
    returns-
        name of trained model
    """
    from fmap2 import FMAP
    # X and Y should be updated if new training data is used
    X=np.load("data/X-tra-new.npy") #training xsec data
    Y=np.load("data/Y-tra-new.npy") #training param data

    fmap=FMAP(model=model,outdir='data')
    fmap.load_data(X,Y)
    model_name = fmap.train(epochs)
    return model_name

def load_fmap(model):
    """
    Loads the trained model.
    """
    from fmap2 import FMAP
    X=np.load("data/X-tra-new.npy") #training xsec data
    Y=np.load("data/Y-tra-new.npy") #training param data
    fmap=FMAP(model=model,outdir='data')
    fmap.load_data(X,Y)
    fmap.load_model()
    return fmap

def plot_loss(models,outdir):
    """
    Plots the loss function of the trained model.

    params-
        models - list of models
        outdir - directory with the saved models
    returns-
        loss plots for each model (.pdf files)
    """
    checkdir('gallery')

    for model in models:
        print(model)
        history=np.load('%s/%s.npy'%(outdir,model), allow_pickle=True).item()
        keys=history.keys()

        loss=history.get('loss')
        print(np.asarray(loss).shape)
        val_loss=history.get('val_loss')
        py.yscale("log")
        py.plot(loss,label=r'$\rm training$')
        py.plot(val_loss,label=r'$\rm validation$')

        py.ylabel(r'$\rm Loss$',size=20)
        py.text(0.2,0.8,r'$\rm %s$'%model,size=20)
        py.legend(loc=1,fontsize=20,frameon=False)
        py.xlabel(r'$\rm Epoch$',size=20)

        py.tight_layout()
        py.savefig('gallery/%s-loss.pdf'%model)
        py.close()

def gen_predictions(model):
    """
    Generates predicted cross section values by a trained model.

    params-
        model - name of trained model
    returns-
        predicted cross section data
    """
    Y=np.load("data/Y-val-2k-new.npy") #validation param data

    fmap=load_fmap(model)
    TY = fmap.ssy.transform(Y)
    TX = fmap.model.predict(TY)
    Xp = fmap.ssx.inverse_transform(TX) #predicted xsecs

    np.save("data/X-pre-%s"%model,Xp,allow_pickle=True)

def analyze_predictions(models,epochs):
    """
    Makes a residual plot of predictions by given models.

    This plot helps see how well our model is predicting as compared to the
    validation samples (theoretical data).

    params-
        models - list of model names
        epochs - epochs the models were trained over (for naming plot)
    """
    X  = np.load("data/X-val-2k-new.npy") #validation xsec data

    nrows, ncols = 1, 1
    fig = py.figure(figsize = (ncols*5, nrows*3))
    ax = py.subplot(nrows, ncols,1)

    for model in models:
        Xp = np.load("data/X-pre-%s.npy"%model,allow_pickle=True) #predicted xsec
        R=(X-Xp)/X #calculates residuals
        R=R.flatten()
        ax.hist(R,range=(-0.02,0.02),bins=100,
                density=True,
                histtype='step',
                label=r'$\rm %s$'%model)

    ax.set_ylabel(r'$\rm Normalized~Yield$',size=20)
    ax.set_xlabel(r'$\rm (x_{\rm val}-x_{\rm pre})/x_{\rm val}$',size=20)
    ax.legend(loc=1,fontsize=6,frameon=False)
    py.tight_layout()
    py.savefig(('gallery/R%s.pdf')%(epochs))

def calc_chi2(models_test):
    """
    Calculates the chi2 value of the given models.

    params-
        models_test - list of model names
    """
    cross_true = np.load('data/X-exp.npy') # true xsec (experimental)
    dx_true = np.load('data/dX-exp.npy') # error in true xsec

    for model in models_test:
        my_cross = np.load('data/X-pre-%s.npy'%model) #predicted xsec
        res = (cross_true - np.mean(my_cross,axis=0))
        res = (res/dx_true)
        chi2 = np.sum(res**2)
        chi2 = chi2/my_cross.shape[1]
        print('model:',model,'chi2:',chi2) #prints chi2 value with model name

def get_distribution(models):
    """
    Creates a histogram of true xsec and predicted xsec.

    params-
        models- list of model names
    """
    X = np.load('data/X-exp.npy')
    nrows, ncols = 1, 1
    fig = py.figure(figsize = (ncols*5, nrows*3))
    ax = py.subplot(nrows, ncols,1)

    # creates the true xsec histogram
    ax.hist(X[0],bins=100,density=True,histtype='step',label="experimental")

    for model in models:
        Xp = np.load("data/X-pre-%s.npy"%model,allow_pickle=True)
        Xp_mean = np.mean(Xp, axis=0) #calculates mean value of each xsec
        # histogram of the mean of each predicted xsec
        ax.hist(Xp_mean,bins=100,density=True,histtype='step',label="%s"%model)

    py.xlabel('X-sec Values')
    py.ylabel('Frequency')
    ax.legend(loc=1,fontsize=6,frameon=False)
    py.tight_layout()
    py.savefig(('gallery/dist-%s.pdf')%(epochs))

def plot_true(models):
    """
    Plots the minimum and maximum values of predicted cross sections along
    with the true cross sections.

    This helps us see the range of the predicted values and whether the true
    values lie within it.

    params-
        models - name of trained model
    """
    nrows, ncols = 1, 1
    fig = py.figure(figsize = (ncols*8, nrows*5))
    ax = py.subplot(nrows, ncols,1)

    true_cross  = np.load('data/X-exp.npy') #true experimental x-sec values
    my_cross = np.load('data/X-pre-%s.npy'%models[0]) #predicted xsec

    min_pred = np.min(my_cross,axis=0) #min value of each predicted xsec
    max_pred = np.max(my_cross,axis=0) #max calue of each predicted xsec

    #array of x-values (index numbers of xsecs)
    x_coord = np.linspace(0,len(true_cross[0]),len(true_cross[0]))

    # creates bands for the min and max predictions
    py.fill_between(x_coord, min_pred, y2=max_pred, where=None,
                    interpolate=False,
                    step=None, alpha=0.3, label='predicted')
    py.plot(true_cross[0], label = 'experimental') # plots true xsec values
    py.ylabel(r'$\rm Cross-section~Values$',size=20)
    py.xlabel(r'$\rm Index~Numbers$',size=20)
    py.legend(loc=1,fontsize=10,frameon=False)
    py.savefig(('gallery/true-%s.pdf'%(epochs)))

if __name__=="__main__":
    models_train=[] #models to train
    models_test =[] #trained models to test

    # e.g.
    models_train.append('ML1-b265-final')
    models_train.append('ML2-b265-final')

    # --train model
    epochs=9000
    for model in models_train:
        models_test.append(train_fmap(model,epochs))

    #models_test.append('ML1-b265-1000-final-3000')
    # --plot losses
    plot_loss(models_test,'./data')

    # --compare predictions
    for model in models_test: gen_predictions(model)

    # manually append model before analyzing
    #models_test.append('ML1-b256-n1000-new-2000')
    #analyze_predictions(models_test,epochs)

    # --calculate chi2
    #calc_chi2(models_test)

    # --plot predicted range
    #plot_true(models_test)

    # --plot distribution of x-sec values
    #get_distribution(models_test)
