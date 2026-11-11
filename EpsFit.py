# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 20:23:17 2021

@author: jmj_
"""
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def EpsOrder2(om,einf,a10,a11,b10,b11,b12,a20,a21,b20,b21,b22):
    eps = einf + ((a10 + a11*om)/(b10 - 1j*b11*om - b12*om**2) +
                  (a20 + a21*om)/(b20 - 1j*b21*om - b22*om**2))
    return np.hstack([eps.real,eps.imag])

def EpsOrder3(om,einf,a10,a11,b10,b11,b12,a20,a21,b20,b21,b22,a30,a31,b30,b31,b32):
    eps = einf + ((a10 + a11*om)/(b10 - 1j*b11*om - b12*om**2) +
                  (a20 + a21*om)/(b20 - 1j*b21*om - b22*om**2) +
                  (a30 + a31*om)/(b30 - 1j*b31*om - b32*om**2))
    return np.hstack([eps.real,eps.imag])

def EpsOrder4(om,einf,a10,a11,b10,b11,b12,a20,a21,b20,b21,b22,a30,a31,b30,b31,b32,a40,a41,b40,b41,b42):
    eps = einf + ((a10 + a11*om)/(b10 - 1j*b11*om - b12*om**2) +
                  (a20 + a21*om)/(b20 - 1j*b21*om - b22*om**2) +
                  (a30 + a31*om)/(b30 - 1j*b31*om - b32*om**2) +
                  (a40 + a41*om)/(b40 - 1j*b41*om - b42*om**2))
    return np.hstack([eps.real,eps.imag])

def EpsOrder2_unstack(om,einf,a10,a11,b10,b11,b12,a20,a21,b20,b21,b22):
    eps = einf + ((a10 + a11*om)/(b10 - 1j*b11*om - b12*om**2) +
                  (a20 + a21*om)/(b20 - 1j*b21*om - b22*om**2))
    return eps

def EpsOrder3_unstack(om,einf,a10,a11,b10,b11,b12,a20,a21,b20,b21,b22,a30,a31,b30,b31,b32):
    eps = einf + ((a10 + a11*om)/(b10 - 1j*b11*om - b12*om**2) +
                  (a20 + a21*om)/(b20 - 1j*b21*om - b22*om**2) +
                  (a30 + a31*om)/(b30 - 1j*b31*om - b32*om**2))
    return eps

def EpsOrder4_unstack(om,einf,a10,a11,b10,b11,b12,a20,a21,b20,b21,b22,a30,a31,b30,b31,b32,a40,a41,b40,b41,b42):
    eps = einf + ((a10 + a11*om)/(b10 - 1j*b11*om - b12*om**2) +
                  (a20 + a21*om)/(b20 - 1j*b21*om - b22*om**2) +
                  (a30 + a31*om)/(b30 - 1j*b31*om - b32*om**2) +
                  (a40 + a41*om)/(b40 - 1j*b41*om - b42*om**2))
    return eps

class EpsFit:
    def __init__(self,mat_file,freq_units,target_model_order=None):
        freq_scale = {'GHz':1e9,'THz':1e12,'PHz':1e15}
        self.lightc = 299792458
        self.fu = freq_units
        self.fs = freq_scale[freq_units]
        
        mat_data = np.loadtxt(mat_file,delimiter=',')
        mat_data = np.swapaxes(mat_data,0,1)
        [freqs,epsr,epsi] = mat_data[0:3]
        epsfit = np.hstack([epsr,epsi])
        freqsPHz = freqs*self.fs*1e-15 #Phz

        if target_model_order is None:
            try:
                p0 = 11*[1]
                p0[0] = 5
                popt, pcov = curve_fit(EpsOrder2, freqsPHz, epsfit,maxfev=100000,p0=p0)
                self.fitorder = 2
            except:
                try:
                    p0 = 16*[1]
                    p0[0] = 5
                    popt, pcov = curve_fit(EpsOrder3, freqsPHz, epsfit,maxfev=100000,p0=p0)
                    self.fitorder = 3
                except:
                    p0 = 21 * [1]
                    p0[0] = 5
                    popt, pcov = curve_fit(EpsOrder4, freqsPHz, epsfit, maxfev=100000, p0=p0)
                    self.fitorder = 4

        elif target_model_order is not None:
            if target_model_order == 2:
                p0 = 11*[1]
                p0[0] = 5
                popt, pcov = curve_fit(EpsOrder2, freqsPHz, epsfit,maxfev=100000,p0=p0)
                self.fitorder = 2
            if target_model_order == 3:
                p0 = 16*[1]
                p0[0] = 5
                popt, pcov = curve_fit(EpsOrder3, freqsPHz, epsfit,maxfev=100000,p0=p0)
                self.fitorder = 3
            if target_model_order == 4:
                p0 = 21 * [1]
                p0[0] = 5
                popt, pcov = curve_fit(EpsOrder4, freqsPHz, epsfit, maxfev=100000, p0=p0)
                self.fitorder = 4
    
        self.popt = popt
        self.fitfreqsPHz = freqsPHz #Phz
        self.fitlam = self.lightc/freqs/self.fs #m
        self.epsdat = epsr + 1j*epsi
        
    def show_fit(self,lamrange=None):
        if lamrange is None:
            freqs = self.fitfreqsPHz
            lam_nm = self.fitlam*1e9
        elif lamrange is not None:
            lam_nm = np.flip(np.linspace(lamrange[0],lamrange[1],500))
            freqs = 1e-15*self.lightc/(lam_nm*1e-9)
            
        if self.fitorder == 2:
            raw_model_eps = EpsOrder2(freqs,*self.popt)
        elif self.fitorder == 3:
            raw_model_eps = EpsOrder3(freqs,*self.popt)
        elif self.fitorder == 4:
            raw_model_eps = EpsOrder4(freqs,*self.popt)
            
        ndat = len(freqs)
        model_eps = raw_model_eps[:ndat] + 1j*raw_model_eps[ndat:]
        fit_eps = self.epsdat
        fit_lam_nm = self.fitlam*1e9
        
        W = 6.72
        H = 0.65*W
        fig, ax = plt.subplots(1,1,figsize=(W,H),dpi=300)
        ax.set_xlim([lamrange[0],lamrange[1]])
        ax.plot(fit_lam_nm,fit_eps.real,linestyle='none',marker='o',ms=3,mew=0.2,fillstyle='none',label='Re[eps] data')
        ax.plot(fit_lam_nm,fit_eps.imag,linestyle='none',marker='o',ms=3,mew=0.2,fillstyle='none',label='Im[eps] data')
        ax.plot(lam_nm, model_eps.real, linestyle='-', lw=0.9, label='Re[eps] fit')
        ax.plot(lam_nm,model_eps.imag,linestyle='-',lw=0.9,label='Im[eps] fit')
        ax.legend()
        ax.grid(ls=':')
        ax.set_xlabel('Wavelength [nm]')
        plt.show()
        
        return

    def show_pretty_fit(self,lamrange=None):
        if lamrange is None:
            freqs = self.fitfreqsPHz
            lam_nm = self.fitlam*1e9
        elif lamrange is not None:
            lam_nm = np.flip(np.linspace(lamrange[0],lamrange[1],500))
            freqs = 1e-15*self.lightc/(lam_nm*1e-9)
            
        if self.fitorder == 2:
            raw_model_eps = EpsOrder2(freqs,*self.popt)
        elif self.fitorder == 3:
            raw_model_eps = EpsOrder3(freqs,*self.popt)
        elif self.fitorder == 4:
            raw_model_eps = EpsOrder4(freqs,*self.popt)
            
        ndat = len(freqs)
        model_eps = raw_model_eps[:ndat] + 1j*raw_model_eps[ndat:]
        fit_eps = self.epsdat
        fit_lam_nm = self.fitlam*1e9
        
        with plt.style.context(['science-arial',{'font.size':7}]):
            textw = 6
            W = textw/2
            H = textw/3

            fig, ax = plt.subplots(1,1,figsize=(W,H),dpi=400)
            ax.plot(fit_lam_nm[::10]*1e-3,fit_eps.real[::10],linestyle='none',marker='o',ms=3,mew=0.3,
                    fillstyle='none',label='Re[Eps] data')
            ax.plot(fit_lam_nm[::10]*1e-3,fit_eps.imag[::10],linestyle='none',marker='o',ms=3,mew=0.3,
                    fillstyle='none',label='Im[Eps] data')
            ax.plot(lam_nm*1e-3,model_eps.real,linestyle='-',lw=0.9,label='Re[Eps] fit')
            ax.plot(lam_nm*1e-3,model_eps.imag,linestyle='-',lw=0.9,label='Im[Eps] fit')
            ax.legend()
            ax.grid(ls=':')
            ax.set_xlabel(r'Wavelength [$\mu$m]')
            ax.set_ylabel('Dielectric function (Eps)')
            ax.set_xticks(np.arange(500,2400+250,250)*1e-3)
            plt.show()
        
        return fig

    def get_eps(self,lamval):
        if isinstance(lamval,float) or isinstance(lamval,int):
            freqvalPHz = 1e-15*self.lightc/(lamval*1e-9)
            if self.fitorder == 2:
                raw_eps = EpsOrder2(freqvalPHz,*self.popt)
            elif self.fitorder == 3:
                raw_eps = EpsOrder3(freqvalPHz,*self.popt)
            elif self.fitorder == 4:
                raw_eps = EpsOrder4(freqvalPHz,*self.popt)
                
            if raw_eps[1] < 0:
                raw_eps[1] = 0
                
            return raw_eps[0] + 1j*raw_eps[1]
        else:
            freqvalPHz = 1e-15*self.lightc/(np.array(lamval)*1e-9)
            if self.fitorder == 2:
                raw_eps = EpsOrder2_unstack(freqvalPHz,*self.popt)
            elif self.fitorder == 3:
                raw_eps = EpsOrder3_unstack(freqvalPHz,*self.popt)
            elif self.fitorder == 4:
                raw_eps = EpsOrder4_unstack(freqvalPHz,*self.popt)
                
            raw_eps[raw_eps.imag < 0.0] = raw_eps[raw_eps.imag < 0.0].real + 0.0j
            return raw_eps

#%%
