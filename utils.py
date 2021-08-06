import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow.keras as keras
import tensorflow as tf

## GENERIC FUNCTIONS

def RK4 (model, v, dt):
    """
    Integrate the model over dt using initial conditions v
    """
    k1 = model(v)
    k2 = model(v + dt*k1/2.)
    k3 = model(v + dt*k2/2.)
    k4 = model(v + dt*k3)
    return v + (1./6.)*dt*(k1 + 2*k2 + 2*k3 + k4)

def rmse (xref, xest, axis=(1,),norm=True):
    """Compute the root mean square. If norm is true (default), the error is normalize by 2 standard deviation."""
    rmse_ret = np.sqrt(np.mean(np.square(xref-xest),axis=axis))
    if norm:
        rmse_ret /= 2*np.std(xref,axis=axis)
    return rmse_ret

def simulate(forward, K, x0, M=None,N=0, burnin=500):
    """
    simulate an ensemble of forward models over K time steps.
    :forward: forward model
    :K: number of time steps to generate
    :N: enseømble size (if N=0, no ensemble)
    :v0: initial state (shape: (3,N)), if N=0. (shape: (3))
    :burnin: number of initial time step to discard (spinup period)
    """
    if M is None:
        M = x0.shape[-1]
    X = np.zeros((K+burnin,N,M)) if N>0 else np.zeros((K+burnin,M))
    X[0,...] = x0
    for i in range(1,K+burnin):
        X[i] =forward(X[i-1])
    return X[burnin:]

def rmse (xref, xest, axis=(1,),norm=True):
    """Compute the root mean square. If norm is true (default), the error is normalize by 2 standard deviation."""
    rmse_ret = np.sqrt(np.mean(np.square(xref-xest),axis=axis))
    if norm:
        rmse_ret /= 2*np.std(xref,axis=axis)
    return rmse_ret

class Periodic1DPadding(keras.layers.Layer):
    """Add a periodic padding to the output of the layer
    (no trainable parameters in this layer)
    # Arguments
        padding_size: tuple giving the padding size (left, right)
    # Output Shape
        input_shape+left+right
    """
    def __init__ (self, padding_size, **kwargs):
        super(Periodic1DPadding, self).__init__(**kwargs)
        if isinstance(padding_size, int):
            padding_size = (padding_size, padding_size)
        self.padding_size = tuple(padding_size)

    def compute_output_shape( self, input_shape ):
        space = input_shape[1:-1]
        if len(space) != 1:
            raise ValueError ('Input shape should be 1D with channel at last')
        new_dim = space[0] + np.sum(self.padding_size)
        return (input_shape[0],new_dim,input_shape[-1])



    def build( self , input_shape):
        super(Periodic1DPadding,self).build(input_shape)

    def call( self, inputs ):
        vleft, vright = self.padding_size
        leftborder = inputs[:, -vleft:, :]
        rigthborder = inputs[:, :vright, :]
        return tf.concat([leftborder, inputs, rigthborder], axis=-2)

## L96 FUNCTIONS

## Inspired from dapper mods/Lorenz96/__init__.py
Force = 8.0 # Forcing
dt = 0.01 # integration time step

def _shift(x, n):
    return np.roll(x, -n, axis=-1)


def L96_dxdt_autonomous(x):
    return (_shift(x, 1) - _shift(x, -2)) * _shift(x, -1) - x

def L96_dxdt(x):
    return L96_dxdt_autonomous(x) + Force

forward_L96 = lambda x: RK4(L96_dxdt, x,dt)

def plot_L96_2D(xx,xxpred,tt,labels,vmin=None,vmax=None,vdelta=None):
    """
    Plot a comparison between two L96 simulations
    """
    if vmin is None:
        vmin,vmax = np.nanmin(xx),np.nanmax(xx)
    if vdelta is None:
        vdelta = np.nanmax(np.abs(xxpred-xx))
    m = xx.shape[1]
    tmin = tt[0]
    tmax = tt[-1]
    fig,ax = plt.subplots(nrows=3,sharex='all')

    divider = [make_axes_locatable(a) for a in ax]

    cax = dict()
    for i in range(3):
        cax [i] = divider[i].append_axes('right', size='5%', pad=0.05)

    delta= dict()
    delta[0] = ax[0].imshow(xx.T,vmin=vmin,vmax =vmax,cmap=plt.get_cmap('viridis'),extent=[tmin,tmax,0,m],aspect='auto')
    delta[1] = ax[1].imshow(xxpred.T,vmin=vmin,vmax=vmax,cmap=plt.get_cmap('viridis'),extent=[tmin,tmax,0,m],aspect='auto')
    delta[2] = ax[2].imshow(xxpred.T- xx.T,cmap=plt.get_cmap('bwr'),
        extent=[tmin,tmax,0,m],aspect='auto',vmin=-vdelta,vmax=vdelta)
    ax[0].set_ylabel(labels[0])
    ax[1].set_ylabel(labels[1])
    ax[2].set_ylabel(labels[1][:2] + ' - ' + labels[0][:2] )
    for i in delta:
        fig.colorbar(delta[i],cax=cax[i],orientation='vertical')
    return fig, ax