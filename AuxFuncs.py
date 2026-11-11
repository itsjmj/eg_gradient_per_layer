import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pathlib
#import pandas as pd
from collections import namedtuple

def open_npz_dict(fname):
    open_data = np.load(fname,allow_pickle=True)
    data = open_data['arr_0'][0]
    open_data.close()
    return data

def save_npz_dict(fname,save_dict):
    np.savez(fname,np.array([save_dict],dtype=object))

def load_listedcmap(cmap_fp):
    cmap = ListedColormap(np.load(cmap_fp))
    return cmap

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
def colorbar(mappable,label=None,ax_list=None,rot=270):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    # try:
    #     cax = divider.append_axes("right", size="5%", pad=0.07)
    # except:
    cax = divider.append_axes('bottom',size="5%", pad=0.07)
    if label is not None:
        if ax_list is not None:
            cb = fig.colorbar(mappable,cax=cax,ax=ax_list)
        else:
            cb = fig.colorbar(mappable,cax=cax)
        cb.ax.get_yaxis().labelpad = 10
        cb.ax.set_ylabel(label, rotation=rot)
    else:
        if ax_list is not None:
            cb = fig.colorbar(mappable,cax=cax,ax=ax_list)
        else:
            cb = fig.colorbar(mappable,cax=cax)
    return cb

from matplotlib.offsetbox import AnchoredText
def add_text_artist(ax,text,loc='upper left',frameon=True):
    at = AnchoredText(text,prop=dict(alpha=0.8),frameon=frameon,loc=loc)
    at.patch.set_alpha(0.6)
    #at.patch.set_boxstyle("alpha=0.9")
    ax.add_artist(at)

def wrap2pi(rad):
    #res = (rad + np.pi) % (2 * np.pi ) - np.pi
    num = (rad + np.pi)
    den = (2 * np.pi )
    return np.remainder(num,den) - np.pi

def wrap22pi(rad):
    num = (rad + 2*np.pi)
    den = (2 * np.pi )
    return np.remainder(num,den) #- np.pi

def angle_deg(c):
    return np.rad2deg(np.angle(c))

def abs_phase_diff(c1,c2,deg = True):
    diff = np.abs(wrap2pi(np.angle(c1) - np.angle(c2)))
    if deg == True:
        return np.rad2deg(diff)
    else:
        return diff
    
def get_GDD_from_S(s,lam):
    '''Assumes wavelength value lambda is in meters, returns GDD in s^2/rad.
       The first value in the array is not accurate, as the difference is taken with respect to the last entry.'''
    s_roll = np.roll(s,1)
    phase_diff = abs_phase_diff(s,s_roll,deg=False)
    abs_phase = np.cumsum(phase_diff)
    omega = 2*np.pi*299792458/lam
    gd = np.gradient(abs_phase,omega)
    gdd = np.gradient(gd,omega)
    return gdd



