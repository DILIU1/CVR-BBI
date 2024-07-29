import os,sys
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPTS_DIR)
sys.path.append(f'{SCRIPTS_DIR}/../')

import socket
from bciBASE.sc import *

from bciBASE.utils import *

from typing import Any
import select
import random
import copy
import time
import datetime
import math
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mne
import pywt
from sklearn.decomposition import FastICA
from statsmodels.tsa.stattools import grangercausalitytests as Granger



class EEG_data:   
    def __init__(self,
                 data: Any = None, 
                 ch_names: Any = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz',
                'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'],
                 sfreq: Any = 1000,
                 trialbaseline: int = 0,
                 minusbaseline: bool = False,
                 fmin: Any = None,
                 fmax: Any = None,
                 nf: Any = None,
                 ch_bad: list = []
                 ):   

        if type(data) == np.ndarray:
            self.data = data
            self.samplelen = self.data.shape[1]
            self.sfreq = sfreq
            self.sampletime = self.samplelen/self.sfreq
            self.ch_names = ch_names
            self.ch_nums = len(self.ch_names)
            self.info = mne.create_info(self.ch_names, self.sfreq, ch_types='eeg')
            self.raw = mne.io.RawArray(self.data, self.info)
        elif type(data) == str:
            if data.endswith('.csv'):
                self.sfreq = sfreq
                self.ch_names = ch_names
                self.ch_nums = len(self.ch_names)
                self.info = mne.create_info(self.ch_names, self.sfreq, ch_types='eeg')
                csv = pd.read_csv(data,header=None)
                # note here that the most recommended way is still to save the csv with only data
                self.data = csv.values[:self.ch_nums,:-1]
                self.samplelen = self.data.shape[1]
                self.sampletime = self.samplelen/self.sfreq
                self.raw = mne.io.RawArray(self.data, self.info)
            elif data.endswith('.npy'):
                self.data = np.load(data)
                self.samplelen = self.data.shape[1]
                self.sfreq = sfreq
                self.sampletime = self.samplelen/self.sfreq
                self.ch_names = ch_names
                self.ch_nums = len(self.ch_names)
                self.info = mne.create_info(self.ch_names, self.sfreq, ch_types='eeg')
                self.raw = mne.io.RawArray(self.data, self.info)
            elif data.endswith('.bdf'):
                self.raw = mne.io.read_raw_bdf(data,preload=True)
                self.info = self.raw.info
                self.sfreq = self.info['sfreq']
                self.ch_names = self.raw.ch_names
                self.ch_nums = len(self.ch_names)
                self.data = self.raw.get_data()
                self.samplelen = self.data.shape[1]
                self.sampletime = self.samplelen/self.sfreq
            elif data.endswith('.edf'):
                self.raw = mne.io.read_raw_edf(data,preload=True)
                self.info = self.raw.info
                self.sfreq = self.info['sfreq']
                self.ch_names = self.raw.ch_names
                self.ch_nums = len(self.ch_names)
                self.data = self.raw.get_data()
                self.samplelen = self.data.shape[1]
                self.sampletime = self.samplelen/self.sfreq
            elif data.endswith('.vhdr'):
                self.raw = mne.io.read_raw(data,preload=True)
                self.info = self.raw.info
                self.sfreq = self.info['sfreq']
                self.ch_names = self.raw.ch_names
                self.ch_nums = len(self.ch_names)
                self.data = self.raw.get_data()
                self.samplelen = self.data.shape[1]
                self.sampletime = self.samplelen/self.sfreq    
        if self.ch_nums != self.data.shape[0]: 
            print(f'Error: chans number dimatch with data!')
            raise         

        if type(trialbaseline)==int:
            self.trialbaseline = trialbaseline
            self.minusbaseline = minusbaseline
            if self.minusbaseline:
                if self.trialbaseline <= self.samplelen and self.trialbaseline > 0:
                    baseline = np.nanpercentile( self.data[:,:self.trialbaseline] , 0.5, axis=1 )
                    self.raw = mne.io.RawArray(self.data - np.repeat(np.array([baseline]),self.samplelen,axis=0).T, 
                                            self.info)
                    print(f'Data baseline {trialbaseline} referenced !')
                else:
                    raise

        if len(ch_bad):
            # ch_bad = [x], x can be 1~ch_nums or ch_names str
            data = self.raw.get_data()
            for ch in ch_bad:
                if type(ch)==int: data[ch-1,:] = np.nan
                elif type(ch)==str: data[self.ch_names.index[ch],:] = np.nan
                else: print(type(ch)); raise
                print(f"change channel {ch} to np.nan")
            self.raw = mne.io.RawArray(data,self.info)

        # note that do not do reference        
        # print(self.raw.get_data().max(axis=1)-self.raw.get_data().min(axis=1))
        # self.raw.set_eeg_reference('average', projection=True)
        # self.raw.apply_proj()
        # print(self.raw.get_data().max(axis=1)-self.raw.get_data().min(axis=1))

        # note that filter flag is to save the filter time
        self.flag_fmin = None
        self.flag_fmax = None
        self.flag_nf = None
        if not (fmin==None and fmax==None):
            # default: method='fir',fir_window='hamming',fir_design='firwin'
            self.raw.filter(fmin,fmax,fir_design='firwin')
            self.flag_fmin = fmin
            self.flag_fmax = fmax
        if not nf==None:
            # note that notch filter should also notch 100/150/200...
            self.raw.notch_filter(nf, fir_design='firwin')  
            self.flag_nf = nf if type(nf)==list else [nf]

        print('Complete initial!\n')
    
    def refilter(self,flag,value,replace=False):
        if value == None: return False
        
        if flag=='fmin':
            if self.flag_fmin==None:
                if replace: self.flag_fmin = value
                return True
            
            if self.flag_fmin<value: 
                if replace: self.flag_fmin = value
                return True
            if self.flag_fmin>=value: return False
        
        if flag=='fmax':
            if self.flag_fmax==None:
                if replace: self.flag_fmax = value
                return True
            
            if self.flag_fmax>value: 
                if replace: self.flag_fmax = value
                return True
            if self.flag_fmax<=value: 
                return False
        
        if flag=='nf':
            if type(value)!=list:
                value = [value]
            if self.flag_nf==None:
                if replace: self.flag_nf = value
                return True
            
            exists = [v not in self.flag_nf for v in value]
            if replace: 
                for exist,v in zip(exists,value):
                    if exist==False: 
                        self.flag_nf.append(v)
            if sum(exists)==False: return False
            else: return True        
           
    def temporal_plot(self,para=None,figfile='',median_plot=False,legend=True,legend_cols=9999,locs=[],fmin=None,fmax=None,nf=None):
        if para==None:
            raw = copy.deepcopy(self.raw)
            if self.refilter('fmin',fmin) or self.refilter('fmin',fmin):
                raw.filter(fmin,fmax,fir_design='firwin')
            if self.refilter('nf',nf):
                raw.notch_filter(nf, fir_design='firwin')  
            data = raw.get_data()
        else:
            # data sequence elf defined as long as the ch_names and ch_nums matches the private 
            data = np.array(para)
        
        l = len(locs)  
        r = 2 if l else 1
       
        fig = plt.figure(figsize=(len(locs),r))
        ax = fig.add_subplot(r,1,1)

        if median_plot == False:
            for i,d in enumerate(data):
                # note that should set colors ourselfs
                ax.plot(d,linewidth=1)
            if legend:
                ncol = min( math.ceil(self.ch_nums/10) , legend_cols)
                plt.legend(self.ch_names,ncol=ncol,loc='upper left',
                           framealpha=1,
                        #    bbox_to_anchor=(0.5,1.2),
                           fontsize=4)
            else:
                plt.legend('',frameon=False)

            if len(locs):
                for i,loc in enumerate(locs):
                    ax = fig.add_subplot(r,len(locs),len(locs)+i+1)
                    if loc>0 and loc<data.shape[1]:
                        self.plot_topo(data[:,loc-1],figfile=None,ax=ax,loc=None,sphere=-1,sensors='none')
                    else:
                        self.plot_topo([np.nan]*self.ch_nums,figfile=None,ax=ax,loc=None,sphere=-1,sensors='none')

        else:
            ch_names_legends = []
            datamedian = np.nanpercentile(data,0.5,axis=1)
            data_median = data - np.repeat(np.array([datamedian]),data.shape[1],axis=0).T
            for d,c,dm in zip(data_median,self.ch_names,datamedian):
                ax.plot(d,linewidth=1)
                ch_names_legends.append(rf'%s: %.1g'%(c,dm))
            if legend:
                ncol = min( math.ceil(self.ch_nums/10) , legend_cols)
                plt.legend(ch_names_legends,ncol=ncol,loc='upper left',
                           framealpha=1,
                           fontsize=4)
            else:
                plt.legend('',frameon=False)
            
            if len(locs):
                for i,loc in enumerate(locs):
                    ax = fig.add_subplot(r,len(locs),len(locs)+i+1)
                    if loc>0 and loc<data.shape[1]:
                        self.plot_topo(data_median[:,loc-1],figfile=None,ax=ax,loc=None,sphere=-1,sensors='none')
                    else:
                        self.plot_topo([np.nan]*self.ch_nums,figfile=None,ax=ax,loc=None,sphere=-1,sensors='none')

        plt.savefig(figfile,dpi=400)
        plt.close()
        print('Done temporary plot!\n')
        

    def compute_psd_features(self, fmin=None, fmax=None, nf=None, picks=[], 
                             return_f='max', snr=False, 
                             delta=[0.01, 4], theta=[4, 8], beta=[8, 14], 
                             alpha=[14, 30], gamma=[30, 100]):
        
        raw = copy.deepcopy(self.raw)
        
        if self.refilter('fmin', fmin) or self.refilter('fmax', fmax):
            raw.filter(fmin, fmax, fir_design='firwin')
        if self.refilter('nf', nf):
            raw.notch_filter(nf, fir_design='firwin')

        fmin = 0.01 if fmin is None else fmin
        fmax = self.sfreq / 2 if fmax is None else fmax

        # Convert channel names to indices
        picks_indices = mne.pick_channels(raw.info['ch_names'], include=picks)

        spectrum = raw.compute_psd('welch',
                                   n_fft=self.samplelen,
                                   n_overlap=0, n_per_seg=None,
                                   tmin=0, tmax=self.sampletime,
                                   fmin=fmin, fmax=fmax,
                                   picks=picks_indices,
                                   window='boxcar', 
                                   verbose=False)
        
        psds, psds_freqs = spectrum.get_data(return_freqs=True)

        # Calculate max frequencies and values
        max1_freqs, max1_values, max2_freqs, max2_values = self.top_two_freqs(psds_freqs, psds, return_f=return_f)
        print(f'Complete PSD calculation and return {return_f}!\n')

        # Calculate SNR if needed
        if snr:
            snrs = snr_spectrum(psds, 
                                noise_n_neighbor_freqs=3,
                                noise_skip_neighbor_freqs=1)
            maxsnr1_freqs, maxsnr1_values, maxsnr2_freqs, maxsnr2_values = self.top_two_freqs(psds_freqs, snrs, return_f=return_f)
            return (max1_freqs, max1_values, max2_freqs, max2_values), (maxsnr1_freqs, maxsnr1_values, maxsnr2_freqs, maxsnr2_values)

        # Calculate rhythm powers
        total_powers = np.nansum(psds, axis=1)
        rhythm_powers = np.zeros((len(picks_indices), 5))
        for ir, rhythm_startend in enumerate([delta, theta, beta, alpha, gamma]):
            start = np.argmin(abs(psds_freqs - rhythm_startend[0]))
            end = np.argmin(abs(psds_freqs - rhythm_startend[1]))
            rhythm_power = psds[:, start:end]
            rhythm_powers[:, ir] = np.nansum(rhythm_power, axis=1) / total_powers
        
        print('Complete rhythm calculation!\n')

        # Combine frequency and amplitude to find the most likely frequency
        most_likely_freq, most_likely_amplitude = self.find_most_likely_freq(max1_freqs, max1_values, max2_freqs, max2_values)

        return (max1_freqs, max1_values, max2_freqs, max2_values), rhythm_powers, (most_likely_freq, most_likely_amplitude)

    def top_two_freqs(self, psds_freqs, psds, return_f='max'):
        sorted_indices = np.argsort(psds, axis=1)[:, ::-1]
        top1_idx = sorted_indices[:, 0]
        top2_idx = sorted_indices[:, 1]

        top1_freqs = psds_freqs[top1_idx]
        top1_values = psds[np.arange(len(psds)), top1_idx]

        top2_freqs = psds_freqs[top2_idx]
        top2_values = psds[np.arange(len(psds)), top2_idx]

        return top1_freqs, top1_values, top2_freqs, top2_values

    def find_most_likely_freq(self, max1_freqs, max1_values, max2_freqs, max2_values, tolerance=1):
        # Combine frequencies and values
        all_freqs = np.hstack((max1_freqs, max2_freqs))
        all_values = np.hstack((max1_values, max2_values))

        # Calculate average amplitude for each unique frequency
        unique_freqs = np.unique(all_freqs)
        avg_amplitudes = [np.mean(all_values[all_freqs == freq]) for freq in unique_freqs]

        # Find the frequency with the maximum average amplitude
        most_likely_idx = np.argmax(avg_amplitudes)
        most_likely_freq = unique_freqs[most_likely_idx]
        most_likely_amplitude = avg_amplitudes[most_likely_idx]
        return most_likely_freq, most_likely_amplitude

    def calc_psd(self,fmin=None,fmax=None,nf=None,
                 picks=[],
                 clac_on_rhythms=False,
                 figfile=None,
                 return_f='max',
                 snr=False,snrfile=None,
                 delta=[0.01,4],theta=[4,8],beta=[8,14],alpha=[14,30],gamma=[30,100]):

        raw = copy.deepcopy(self.raw)
        if self.refilter('fmin',fmin) or self.refilter('fmin',fmin):
            raw.filter(fmin,fmax,fir_design='firwin')
        if self.refilter('nf',nf):
            raw.notch_filter(nf, fir_design='firwin')  

        if fmin==None: 
            fmin = 0.01 if self.flag_fmin==None else self.flag_fmin
        if fmax==None: 
            fmax = self.sfreq/2 if self.flag_fmax==None else self.flag_fmax  
        
        spectrum = raw.compute_psd('welch',
                                        n_fft=self.samplelen,
                                        n_overlap=0, n_per_seg=None,
                                        tmin=0, tmax=self.sampletime,
                                        fmin=fmin, fmax=fmax,
                                        picks=picks,
                                        window='boxcar', 
                                        verbose=False
                                        #,average='mean'
                                        )
        psds, psds_freqs = spectrum.get_data(return_freqs=True)
        
        if not clac_on_rhythms:

            max_freqs,max_freqs_values = max_freq(psds_freqs, psds, return_f=return_f)
            print(f'Complete psd clac and return {return_f}!\n')
            if figfile:
                pn = len(picks)
                fig = plt.figure(figsize=(5,pn*2))
                for iax, ch in enumerate(picks):
                    # ax = fig.add_subplot(2*pn,1,2*iax+1)
                    ax = fig.add_subplot(pn,1,iax+1)
                    ax.plot(psds_freqs, psds[iax,:])
                    ax.set_title(f'{ch} psds')
                    # ax = fig.add_subplot(2*pn,1,2*iax+2)
                    # ax.plot(psds_freqs, snrs[iax,:])
                    # ax.set_title(f'{ch} snrs')
                fig.tight_layout()
                fig.savefig(figfile)
                plt.close()
            
            if snr==False or snrfile==None:
                return max_freqs,max_freqs_values
            else:
                snrs = snr_spectrum(psds, 
                                noise_n_neighbor_freqs=3,
                                noise_skip_neighbor_freqs=1)
                pn = len(picks)
                fig = plt.figure(figsize=(5,pn*2))
                for iax, ch in enumerate(picks):
                    # ax = fig.add_subplot(2*pn,1,2*iax+1)
                    ax = fig.add_subplot(pn,1,iax+1)
                    ax.plot(psds_freqs, snrs[iax,:])
                    ax.set_title(f'{ch} snrs')
                    # ax = fig.add_subplot(2*pn,1,2*iax+2)
                    # ax.plot(psds_freqs, snrs[iax,:])
                    # ax.set_title(f'{ch} snrs')
                fig.tight_layout()
                fig.savefig(snrfile)
                plt.close()
                print(f'Complete snr clac and return {return_f}!\n')
                maxsnr_freqs,max_snrs_values = max_freq(psds_freqs, snrs, return_f=return_f)
                return max_freqs,max_freqs_values,maxsnr_freqs,max_snrs_values
        
        else:
            pn = len(picks)
            total_powers = np.nansum(psds,axis=1)
            rhythm_powers = np.zeros((pn,5))
            fn = len(psds_freqs)
            for ir,rhythm_startend in enumerate([delta,theta,beta,alpha,gamma]):
                start = np.argmin(abs(psds_freqs-rhythm_startend[0]))
                end = np.argmin(abs(psds_freqs-rhythm_startend[1]))
                if end+1 == fn: end += 1
                rhythm_freq = psds_freqs[start:end]
                rhythm_power = psds[:,start:end]
                rhythm_powers[:,ir] = np.nansum(rhythm_power,axis=1)/total_powers
            print('Complete rhythm clac!\n')
            return rhythm_powers
        
    

    def calc_features(self,feature_names=['len','freq'],
                      fmin=None,fmax=None,nf=None,
                      calc_median=False, 
                      remove_baseline=False,
                      figfile=None,
                      figr='auto'):
        if len(feature_names)==0: return

        raw = copy.deepcopy(self.raw)
        if self.refilter('fmin',fmin) or self.refilter('fmin',fmin):
            raw.filter(fmin,fmax,fir_design='firwin') 
        if self.refilter('nf',nf):
            raw.notch_filter(nf,fir_design='firwin') 
        data = raw.get_data()
        
        if calc_median:
            datamedian = np.nanpercentile(data,0.5,axis=1)
            data = data - np.repeat(np.array([datamedian]),self.samplelen,axis=0).T

        if remove_baseline:
            # only right when already minus baseline 
            data = data[:,self.trialbaseline:]            

        df_features =  pd.DataFrame([])
        df_features.index = self.ch_names    
        for feature_name in feature_names:
            if feature_name == 'len':
                feature = np.array([self.samplelen]*self.ch_nums)        
            if feature_name == 'freq':
                feature = np.array([self.sfreq]*self.ch_nums)
            if feature_name == 'baseline':
                feature = self.trialbaseline
            if feature_name == 'amplitude':
                feature = np.nanmax(data,axis=1) - np.nanmin(data,axis=1)
            if feature_name == 'latency_max':
                # /s
                # feature = np.argsort(data,axis=1)[:,-int(self.sfreq*0.2):].mean(axis=1)
                feature = np.nanargmax(data,axis=1) / self.sfreq
            if feature_name == 'latency_min':
                # /s
                # feature = np.argsort(data,axis=1)[:,:int(self.sfreq*0.2)].mean(axis=1)
                feature = np.nanargmin(data,axis=1) / self.sfreq
            if feature_name == 'latency_extremum':
                # /s
                datamedian = np.nanpercentile(data,0.5,axis=1)
                datamax = np.nanmax(data,axis=1)
                lmaxs = np.nanargmax(data,axis=1) / self.sfreq
                datamin = np.nanmin(data,axis=1)
                lmins = np.nanargmin(data,axis=1) / self.sfreq
                feature = [lmax if dmax-dm<dm-dmin else lmin for lmax,lmin,dmax,dmin,dm in zip(lmaxs,lmins,datamax,datamin,datamedian)]
            if feature_name == 'std':
                feature = np.nanstd(data,axis=1)
            if feature_name == 'skew':
                feature = stats.skew(data,axis=1)
            if feature_name == 'kurtosis':
                feature = stats.kurtosis(data,axis=1)
            if feature_name == 'de10':
                data_des = []
                de_time = self.sfreq/10
                de_step = max( 10, math.floor(data.shape[1]/de_time) )
                for i in range(self.ch_nums):
                    for j in range(0,data.shape[1],de_step):
                        d = data[i,j:j+1000]
                        d = d - np.nanmin(d)
                        entropy = stats.entropy(d)
                        data_des.append(entropy)
                data_des = (np.array(data_des)).reshape(self.ch_nums,-1)
                
                # todel
                print(data_des.shape)

                feature = np.nanmean(data_des,axis=1)
            if feature_name == 'zcr':
                # 0~1
                datamedian = np.nanpercentile(data,0.5,axis=1)
                dm = data - np.repeat(np.array([datamedian]),data.shape[1],axis=0).T
                zc = (dm[:,:-1] * dm[:,1:] < 0)
                feature = zc.sum(axis=1)/data.shape[1]
            if feature_name == 'olr': # outliner ratio
                # 0~1
                datamean = np.nanmean(data,axis=1)
                dm = data - np.repeat(np.array([datamean]),data.shape[1],axis=0).T
                datastd = np.nanstd(data,axis=1)
                ds = np.repeat(np.array([datastd]),data.shape[1],axis=0).T
                ol = abs(dm) > ds
                feature = ol.sum(axis=1)/data.shape[1]
            if feature_name == 'peak': # deleloped by binghua
                # / number in 1s
                peaks_ratio = []
                valleys_ratio = []
                for i in range(self.ch_nums):
                    x_peak = find_mypeaks(data[i,:], fp=True,method='AMPD',figfile=None,ax=None)
                    x_valley= find_mypeaks(data[i,:], fp=False,method='AMPD',figfile=None,ax=None)
                    peaks_ratio.append( self.sfreq * len(x_peak) / data.shape[1] )
                    valleys_ratio.append( self.sfreq * len(x_valley) / data.shape[1] )              
                feature = [i+j for i,j in zip(peaks_ratio,valleys_ratio)]
            if feature_name == 'rhythm':
                # 0~1
                delta=[0.01,4]
                theta=[4,8]
                beta=[8,14]
                alpha=[14,30]
                gamma=[30,100]
                rhythm_powers = self.calc_psd(fmin=fmin,fmax=fmax,nf=nf,
                       picks=self.ch_names, 
                       clac_on_rhythms=True)
                df_features[f'{delta}power'] = rhythm_powers[:,0]
                df_features[f'{theta}power'] = rhythm_powers[:,1]
                df_features[f'{beta}power'] = rhythm_powers[:,2]
                df_features[f'{alpha}power'] = rhythm_powers[:,3]
                df_features[f'{gamma}power'] = rhythm_powers[:,4]
                continue
            df_features[feature_name] = feature
        print(f'Complete feature clac!\n')

        if figfile:
            col = df_features.columns
            f = len(col)
            if figr=='auto':
                h = int(math.sqrt(f))
                w = math.ceil(f/h)
            else:
                h = 1
                w = f
            fig = plt.figure(figsize=(4*w,4*h))
            for iax,c in enumerate(col):
                ax = fig.add_subplot(h,w,iax+1)
                scale = self.plot_topo(para=df_features[c].values,scale=df_features.values.max(),figfile=None,ax=ax,loc=None,sphere=-1,sensors='none')
                ax.set_title('feature: %s scale: %.1g'%(c,scale))
            plt.tight_layout()
            plt.savefig(figfile,dpi=400)
            plt.close()

        return df_features

    def plot_topo(self,para,scale=None,figfile=None,ax=None,loc=None,sphere=-1,sensors='name'):
        # data should be 1 dimen and has same len with ch_names
        data = np.array(para)

        # normalization
        # data = (data - data.min()) / (data.max() - data.min()) # old way
        if scale==None:
            scale = np.nanmax(abs(data))
        if scale!=scale or scale==0: 
            scale=np.nan
        else:
            data[data!=data] = 0
        data = data / scale

        if type(loc) != pd.core.frame.DataFrame:
            pd20 = pd.read_csv(f'{SCRIPTS_DIR}/montage.csv',index_col=0)
            pd32 = pd20.loc[self.ch_names]
        else:
            # if loc given outside,
            # this topo has noothing to do with the private variables
            pd32 = loc.copy()
        
        if sphere==-1:
            sphere = 0.11
        if data.shape[0] < 32 :
            extrapolate = 'local'
        else:
            extrapolate = 'head'

        if sensors=='name':
            names = pd32.index
        elif sensors=='none':
            names = None
        elif sensors=='number':
            names = list(range(1,pd32.shape[0]+1))

        if figfile:
            fig = plt.figure(figsize=(2,2))
            axes = fig.add_subplot(1,1,1)
            mne.viz.plot_topomap(data,
                                pos = pd32[['RIGHT','NOSE']].values,
                                sensors=True,
                                names = names,
                                contours=10, 
                                outlines='head',
                                sphere=sphere, 
                                extrapolate=extrapolate,
                                border='mean',
                                cmap = 'RdBu_r', 
                                vlim = [-1,1],
                                show = False,
                                axes = axes)
            plt.tight_layout()
            plt.savefig(figfile,dpi=400)
            plt.close()

        if ax: 
            mne.viz.plot_topomap(data,
                                pos = pd32[['RIGHT','NOSE']].values,
                                sensors=True,
                                names = names,
                                contours=10, 
                                outlines='head',
                                sphere=sphere, 
                                extrapolate=extrapolate,
                                border='mean',
                                cmap = 'RdBu_r',
                                vlim = [-1,1], 
                                show = False,
                                axes = ax)
        
        return scale

    def plot_topo_old(self,para,figfile:str):
        pd20 = pd.read_csv(f'{SCRIPTS_DIR}/montage.csv',index_col=0)
        pd32 = pd20.loc[self.ch_names]

        if type(para)==int:
            # time domain samle point
            scale = int(1 / abs(self.data[:,para]).min())
            data = (self.data[:,para]*scale).round().astype(int)
        else:
            # data sequence by chs
            scale = int(1 / abs(np.array(para)).min()) 
            data = (np.array(para)*scale).round().astype(int)
        shift = abs(data).max()
        data = data + shift
        pd_plot = pd32.copy()
        pd_plot['value'] = 1
        for iter,ch in enumerate(pd32.index):
            d = max( 1, round(data[iter]) )
            if d>1:
                pd_ = pd32.loc[[ch]*(d-1)]
                pd_plot = pd.concat([pd_plot,pd_])
                pd_plot.loc[ch,'value'] = d
        print(pd_plot)
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(1,1,1)
        # import seaborn as sns
        # sns.palplot(sns.color_palette('coolwarm'))
        # /colors = plt.cm.get_cmap('tab20').colors
        sns.kdeplot(x=pd_plot['RIGHT'],y=pd_plot['NOSE'],cmap='coolwarm',ax=ax)
        sns.scatterplot(pd32,x='RIGHT',y='NOSE',c='black',ax=ax)
        for ch,row in pd32.iterrows():
            ax.text(row['RIGHT'],row['NOSE'],ch,c='black',fontsize=8)
        plt.tight_layout()
        fig.savefig(figfile)
        plt.close()



    def TimeFrequencyCWT(self,picks:list,
                                   returns = [],
                                   wavename='cgau8',
                                   waveinter=0.1,
                                   figfile=None,
                                   fmin=None,
                                   fmax=None,
                                #    nf=None,
                                   maskTarget=[],
                                   mask=[],
                                   calc_on_rhythms=False,
                                   delta=[0.01,4],theta=[4,8],beta=[8,14],alpha=[14,30],gamma=[30,100]
                                    ):
        _returns = []

        raw = copy.deepcopy(self.raw)
        # if self.refilter('fmin',fmin) or self.refilter('fmin',fmin):
        #     raw.filter(fmin,fmax,fir_design='firwin') 
        # if self.refilter('nf',nf):
        #     raw.notch_filter(nf,fir_design='firwin') 
        raw_data = raw.get_data()

        for p,ch in enumerate(picks):
            i = self.ch_names.index(ch)
            data = raw_data[i,:]

            fc = pywt.central_frequency(wavename)
            totalscal = self.sfreq / 2 / waveinter
            cparam = 2 * fc * totalscal
            scales = cparam / np.arange(totalscal, 0, -1)
            [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / self.sfreq)
            t_squeeze = np.arange(self.samplelen)/self.sfreq

            # maskTarget: [], 'time' or 'frequency'
            # mask: [start,end], /s or /Hz, 0~self.sampletime or 0~self.sfreq/2
            if maskTarget=='time':
                start,end = mask
                it_former = np.argmin(abs(t_squeeze-start))
                it_latter = np.argmin(abs(t_squeeze-end))
                cwtmatr[:,it_former:it_latter+1] = 0
            if maskTarget=='frequency':
                start,end = mask
                if_latter = np.argmin(abs(frequencies-start))
                if_former = np.argmin(abs(frequencies-end))
                cwtmatr[if_former:if_latter+1,:] = 0
            
            if figfile:
                if p==0:
                    lp = len(picks)
                    w = max( 3, int(raw_data.shape[1]/6000) )
                    fig = plt.figure(figsize=(w,lp))

                if fmin==None: 
                    fmin = 0.01 if self.flag_fmin==None else self.flag_fmin
                if fmax==None: 
                    fmax = self.sfreq/2 if self.flag_fmax==None else self.flag_fmax                
                iplot_latter = np.argmin(abs(frequencies-fmin))
                iplot_former = np.argmin(abs(frequencies-fmax))
                plotfrequencies = frequencies[iplot_former:iplot_latter+1]
                plotmatr = cwtmatr[iplot_former:iplot_latter+1,:]

                ax = fig.add_subplot(lp,1,p+1)
                ax.contourf(t_squeeze, plotfrequencies, abs(plotmatr)
                            # ,vmax=vmax,vmin=0
                            )
                ax.set_ylabel(u"Y-Hz", fontsize=10/lp)
                ax.set_xlabel(u"X-S", fontsize=10/lp)
                ax.set_title(ch)

                if p==len(picks)-1:
                    plt.subplots_adjust(hspace=0.4)
                    plt.tight_layout()
                    plt.savefig(figfile,dpi=400)
                    plt.close() 
            
            if 'matrix' in returns:
                if p==0:
                    matrix_dict = {}
                df = pd.DataFrame(cwtmatr)
                df.index = frequencies
                df.columns = t_squeeze 
                matrix_dict[ch] = df
                if p==len(picks)-1:
                    _returns.append(matrix_dict)

            if 'center' in returns:  
                if p==0: 
                    centers_dict = {}
                df = pd.DataFrame(abs(cwtmatr))
                df.index = frequencies
                df.columns = t_squeeze 
                if calc_on_rhythms: bands=[delta,theta,beta,alpha,gamma]
                else: bands=[[fmin,fmax]]
                for ir,rhythm_startend in enumerate(bands):
                    irf_latter = np.argmin(abs(frequencies-rhythm_startend[0]))
                    irf_former = np.argmin(abs(frequencies-rhythm_startend[1]))
                    if  irf_latter+1 == len(frequencies):  irf_latter += 1
                    df_r = df.loc[ frequencies[ irf_former: irf_latter] ]
                    df_r.to_csv('text.csv')
               
                center = None
                centers_dict[ch] = center
                # note to develop methods to change the return to the frequency centers
                if p==len(picks)-1:
                    _returns.append(centers_dict)
        
        print(f'Complete cwt!\n')
        return _returns
    


    def calc_projection(self,methods='plv',
                          mparas=None,
                          calc_median=False,
                          remove_baseline=True,
                          figfile=None):
        """
        Parameters
        ----------
        methods: tLCS / fLCS / plv / Granger
        mparas: if methods=='tLCS':

        Returns
        -------
        values: DataFrame, element e(i,j) means i data is e timepoints faster than j data 
        """
        print(f'\n{methods}')
        data = self.raw.get_data()

        if calc_median:
            datamedian = np.nanpercentile(data,0.5,axis=1)
            data = data - np.repeat(np.array([datamedian]),self.samplelen,axis=0).T

        if remove_baseline:
            # only right when already minus baseline 
            data = data[:,self.trialbaseline:] 

        values = pd.DataFrame([],index = self.ch_names,columns = self.ch_names)
        if methods=='tLCS':
            gap = int(0.01 * self.sfreq)
            for id in range(self.ch_nums):
                for jd in range(self.ch_nums):
                    idata = data[id,:]
                    jdata = data[jd,:]
                    inormalized = (idata[gap:] - idata[:-gap])[::gap] > 0
                    jnormalized = (jdata[gap:] - jdata[:-gap])[::gap] > 0
                    _,lenLCS,_,posi,posj = longestCommonSubsequence(inormalized, jnormalized)
                    if lenLCS: # note to change this condition
                        value = (posj[0] - posi[0]) * gap / self.sfreq
                    else:
                        value = np.nan
                    values.loc[values.index[id],values.columns[jd]] = value
        elif methods=='plv': 
            for id in range(self.ch_nums):
                for jd in range(self.ch_nums):
                    idata = data[id,:]
                    jdata = data[jd,:]
                    phase_channel1 = np.angle(np.fft.fft(idata))
                    phase_channel2 = np.angle(np.fft.fft(jdata))
                    phase_difference = phase_channel1 - phase_channel2
                    mean_phase_difference = np.mean(np.exp(1j * phase_difference))
                    plv = np.abs(mean_phase_difference)
                    values.loc[values.index[id],values.columns[jd]] = plv
        elif methods=='fLCS':
            _ = self.plot_TimeFrequencyCWT(picks=self.ch_names,
                                   figfile=None,
                                   returns = 'matrix'
                                    )
            matrix_dict = _[0]
            for ic in self.ch_names:
                for jc in self.ch_names:
                    id = matrix_dict[ic]
                    jd = matrix_dict[jc]
                    print(id,jd)
                    raise
                    # note to be continue

        else:
            raise
        return values

    def calc_topo_bySKLEARN(self,para=None,return_loc=True,fmin=5,fmax=20,nf=50):
        # data should be 1 dimen and has same len with ch_names
        if para==None:
            raw = copy.deepcopy(self.raw)
            if self.refilter('fmin',fmin) or self.refilter('fmin',fmin):
                raw.filter(fmin,fmax,fir_design='firwin')
            if self.refilter('nf',nf):
                raw.notch_filter(nf, fir_design='firwin')  
            data = raw.get_data()
        else:
            # data sequence by chs
            data = np.array(para)
            
        
        # x = SR * s
        # SR = x * s-1
        # s = SR-1 * x
        fast_ica = FastICA(n_components=data.shape[0]) 
        Sr = fast_ica.fit_transform(data) # row m of Sr: for chan m of x, weight of every source

        if return_loc:
            Sr_1 = np.linalg.pinv(Sr) # row n of Sr_1: for component n of s, weight of every chan 
            # s = np.dot(Sr_1, data)
            weight = pd.DataFrame(abs(Sr_1)).div(abs(Sr_1).max(axis=1), axis=0)
            pd20 = pd.read_csv(f'{SCRIPTS_DIR}/montage.csv',index_col=0)
            pd32 = pd20.loc[self.ch_names]
            loc = pd.DataFrame(np.dot(weight , pd32.values))
            loc.index = [f's{i+1}' for i in range(pd32.shape[0])]
            loc.columns = pd32.columns
            return Sr, loc
        else:
            return Sr
        
    def calc_topo_byMNE(self):
        # note too develop by binghua
        return



class Room: 
    def __init__(self,roomID):
        self.roomID=roomID
        self.room_times=1
        self.max_room_times=2
        self.master=''
        self.members_info = {}  # str: {}
        self.members_data = {}  # str: []
        self.members_marker = {}  # str: []
        self.members_outcome = {} # str: []
        self.benchmark = []
        self.majority_outcome = []
        self.average_outcome = []
        self.source_outcome = [] 
        self.t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    def save_data(self):
        pass

    def print_room(self):
        print(f"Time: {self.room_times}\n   "+
             f"Master sock: {self.master}\n   "+
             f"Members info: {self.members_info}!\n   "+
             f"Members data markers: {self.members_marker}\n   "+
             f"embers data benchmark: {self.benchmark}\n   "+
             f"Members data outcome: {self.members_outcome}\n   "+
             f"All Members majority voting outcome: {self.majority_outcome}\n   "+
             f"All Members average voting outcome: {self.average_outcome}\n   "+
             f"All Members source voting outcome: {self.source_outcome}\n   "
             )

    def del_member(self,SOCK):
        sock = str(SOCK.getpeername())
        if sock in room.members_info.keys():
            
            # NOTE TO add data saving process 
            # NOTE data saving while member offline
            self.save_data()

            self.members_info.pop(sock)
            self.members_data.pop(sock)
            self.members_marker.pop(sock)
            self.members_outcome.pop(sock)
            
            if( len(self.members_info)>0 and self.master==sock ):
                self.change_room_master
            
    def change_room_master(self):
        self.master = self.members_info.keys()[0]
        print(f"The master sock of Room{self.roomID} should change to {self.master}!")
        self.broadcast_massage('serverbroadcastmaster',[self.master])

    def broadcast_massage(self,messageid,message):
        # message: list
        print(f"Room{self.roomID} broadcasting message to all members!")
        EncodeMessage = Message('encode',self.roomID,messageid,message)
        meg = EncodeMessage.message
        for member in self.members_info.keys():
            SOCK = self.members_info[member]['SOCK']
            SOCK.send(meg)
            print(f"Message send to {member}!")
        print('\n')

    def add_member(self, SOCK, content):
        # set master sock and enable master related 
        sock = str(SOCK.getpeername())
        if len(self.members_info)==0:
            self.master = sock
        
        EncodeMessage = Message('encode',self.roomID,'serverbroadcastmaster',[self.master])
        meg = EncodeMessage.message
        SOCK.send(meg)

        # add member into members_info
        if sock not in self.members_info.keys():
            self.members_info[sock] = {}

            self.members_info[sock]['SOCK'] = SOCK
            for k,v in content[0].items():
                self.members_info[sock][k] = v
            self.members_info[sock]['PARADIGMPARA'] = content[1]
            print(f"Room{self.roomID} and sock {sock} add member info record {self.members_info[sock]}!")

            if not sock == self.master:
                if not ( self.members_info[sock]['paradigm'] == self.members_info[self.master]['paradigm'] and
                  self.members_info[sock]['PARADIGMPARA'][0] == self.members_info[self.master]['PARADIGMPARA'][0] and
                  self.members_info[sock]['trialschan'] == self.members_info[self.master]['trialschan'] and
                  self.members_info[sock]['trialsfreq'] == self.members_info[self.master]['trialsfreq'] ): 
                    raise
            
            # set member related other variables
            self.members_data[sock] = []
            self.members_marker[sock] = [0,0]
            self.members_outcome[sock] = []

    def gen_startpara_for_SSVEPQC(self):      
        
        # NOTE the server set 10 and 13hz here for ssvep paradigm
        ssvep_freqs = [10,13] 
        if len(self.members_info[self.master]['PARADIGMPARA'])==1:    
            self.members_info[self.master]['PARADIGMPARA'].append(ssvep_freqs)

        random_idx = random.randint(0,self.members_info[self.master]['PARADIGMPARA'][0]-1)
        print(f"Generated rand index {random_idx} for SSVEP_QC client!")
        
        self.broadcast_massage('serverbroadcaststart',[random_idx, ssvep_freqs])              

    def gen_hintpara_for_SSVEPQC(self):
        random_idx = random.randint(0,self.members_info[self.master]['PARADIGMPARA'][0]-1)
        print(f"Generated rand index {random_idx} for SSVEP_QC client!")
        
        ssvep_freqs = self.members_info[self.master]['PARADIGMPARA'][1]
        bench = ssvep_freqs[self.benchmark[-1]] 
        majority = self.majority_outcome[-1]
        average = self.average_outcome[-1]
        source = self.source_outcome[-1]
        majority_correctness = 'majority_correct' if majority==bench else 'majority_incorrect'
        average_correctness = 'average_correct' if average==bench else 'average_incorrect'
        source_correctness = 'source_correct' if source==bench else 'source_incorrect'        
        for sock in self.members_info.keys():
            classify = self.members_outcome[sock][-1]
            correctness = 'correct' if classify==bench else 'incorrect'
            majority_matchness = 'match_majority' if classify==bench else 'dismatch_majority'
            average_matchness = 'match_average' if classify==bench else 'dismatch_average'
            source_matchness = 'match_source' if classify==bench else 'dismatch_source'
            log = f'trial,{self.room_times},benchmark,{bench},psd,{classify},correctness,{correctness},' + \
                  f'majority,{majority},majority_matchness,{majority_matchness},majority_correctness,{majority_correctness},' + \
                  f'average,{average},average_matchness,{average_matchness},average_correctness,{average_correctness},' + \
                  f'source,{source},source_matchness,{source_matchness},source_correctness,{source_correctness}\n'

            SOCK = self.members_info[sock]['SOCK']
            EncodeMessage = Message('encode',self.roomID,'serverbroadcasthint',
                                    [random_idx, log])
            meg = EncodeMessage.message
            SOCK.send(meg)  
            print(f"Message send to {sock}!")
        print('\n')
 
    def gen_lastpara_for_SSVEPQC(self):
        ssvep_freqs = self.members_info[self.master]['PARADIGMPARA'][1]
        benchmark = [ssvep_freqs[i] for i in self.benchmark] # from 0/1 to 10/13
        majority = self.majority_outcome
        average = self.average_outcome
        source = self.source_outcome
        majority_correctness = 'majority_correct' if majority[-1]==benchmark[-1] else 'majority_incorrect'
        average_correctness = 'average_correct' if average[-1]==benchmark[-1] else 'average_incorrect'
        source_correctness = 'source_correct' if source[-1]==benchmark[-1] else 'source_incorrect'     
        majority_num_correct = (np.array(majority)==benchmark).sum()
        average_num_correct = (np.array(average)==benchmark).sum()
        source_num_correct = (np.array(source)==benchmark).sum()
        for sock in self.members_info.keys():
            classify = self.members_outcome[sock]
            correctness = 'correct' if classify[-1]==benchmark[-1] else 'incorrect'
            majority_matchness = 'match_majority' if classify[-1]==benchmark[-1] else 'dismatch_majority'
            average_matchness = 'match_average' if classify[-1]==benchmark[-1] else 'dismatch_average'
            source_matchness = 'match_source' if classify[-1]==benchmark[-1] else 'dismatch_source'
            log0 = f'trials,{self.room_times},benchmark,{benchmark[-1]},psd,{classify[-1]},correctness,{correctness},' + \
                  f'majority,{majority[-1]},majority_matchness,{majority_matchness},majority_correctness,{majority_correctness},' + \
                  f'average,{average[-1]},average_matchness,{average_matchness},average_correctness,{average_correctness},' + \
                  f'source,{source[-1]},source_matchness,{source_matchness},source_correctness,{source_correctness}\n'
            classify_num_correct = (np.array(classify)==benchmark).sum()
            match_num_majority = (np.array(majority)==classify).sum()
            match_num_average = (np.array(average)==classify).sum()
            match_num_source = (np.array(source)==classify).sum()
            log1 = f'trials_num,{self.room_times},benchmarks,{benchmark},psds,{classify},correctness_num,{classify_num_correct},' + \
                   f'majoritys,{majority},majority_matchness_num,{match_num_majority},majority_correctness_num,{majority_num_correct},' + \
                   f'averages,{average},average_matchness_num,{match_num_average},average_correctness_num,{average_num_correct},' + \
                   f'sources,{source},source_matchness_num,{match_num_source},source_correctness_num,{source_num_correct}\n'

            SOCK = self.members_info[sock]['SOCK']
            EncodeMessage = Message('encode',self.roomID,'serverbroadcastoutcome',
                                    [log0,log1])
            meg = EncodeMessage.message
            SOCK.send(meg)  
            print(f"Message send to {sock}!")
        print('\n')
            
    def process_data(self):
        one_outs = []
        ch_outs = []
        datas = []
        paradigm = self.members_info[self.master]['paradigm']
        channels = self.members_info[self.master]['trialschan']
        channelnums = len(channels)
        sample = self.members_info[self.master]['trialsfreq']
        datalen = self.members_marker[self.master][-2] - self.members_marker[self.master][-3]
        for sock,clientdata in self.members_data.items(): 
            data = clientdata[-datalen:]
            data = np.reshape(data,(-1,channelnums)).T 
            datas.append(data)

            if paradigm=='SSVEP_QC':
                _EEG_data = EEG_data(data,
                                     channels,
                                     channelnums,
                                     sample,
                                     )
                # _EEG_data.calc_stat()
                
                ssvep_freqs = self.members_info[self.master]['PARADIGMPARA'][-1]
                picks=['O1','Oz','O2']
                max_freqs,_ = _EEG_data.calc_psd(picks)
                ch_outs.extend(max_freqs)
                
                _,max_freq = ssvep_judge(reversed(max_freqs),ssvep_freqs)
            self.members_outcome[sock].append(max_freq)
            one_outs.append(max_freq)
        self.majority_outcome.append(np.unique(one_outs)[-1])

        if paradigm=='SSVEP_QC':
            _,clients_out = ssvep_judge(reversed(ch_outs),ssvep_freqs)
            self.average_outcome.append(clients_out)
        
        self.source_outcome.append(-1)


        # _EEG_data = EEG_data()
            
        # data = list(self.members_data.values())

        # aout = _EEG_data.AverageVote(data)
        # self.average_outcome.append(aout)
        
        # sout = _EEG_data.SourceVote(data)
        # self.source_outcome.append(sout)

    def add_data(self, SOCK, content):
        sock = str(SOCK.getpeername())
        self.members_data[sock].extend(content)
        self.members_marker[sock][-1] = self.members_marker[sock][-1] + len(content)    

    def add_dataover(self, SOCK, content):
        sock = str(SOCK.getpeername())
        self.members_marker[sock].append(self.members_marker[sock][-1])
        
        # check data len mismatch
        datalen, databench = content
        if not datalen == self.members_marker[sock][-2] - self.members_marker[sock][-3]: raise
        if len(self.benchmark) == self.room_times:
            if self.benchmark[-1] != databench: raise
        else:
            self.benchmark.append(databench)



if __name__ == '__main__':
    # create main server socket
    ADDR = (HOST, PORT)
    _service_socket = socket.socket() 
    _service_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # bind server socket
    _service_socket.bind(ADDR) 
    # set max connection for this server socket
    _service_socket.listen(10) 
    # initilize server sockets list and connected rooms dict
    _current_in_list = [_service_socket]
    _rooms = dict()
    print( f'\nStart server service, create main socket {str(_service_socket.getsockname())}, and keep listenning coonection...\n' )
    while True:
        # read list \ write list \ exception list
        rlist, wlist, xlist = select.select(_current_in_list, [], [])
        for SOCK in rlist:


            # if main socket
            if SOCK is _service_socket:
                # accept: block until client connect, 
                client, addr = SOCK.accept()
                # accepted client will be next sock and handle
                _current_in_list.append(client)
                print(f"\nClient (%s: %s) connected to main server sock!\n" % addr)
            

            # if answer socket
            else:
                try:
                    raw_message = SOCK.recv(BUFSIZE)
                    if raw_message:
                        
                        DecodeMessage = Message('decode',raw_message)
                        message = DecodeMessage.message
                        room_id = DecodeMessage.room_id
                        messageid = DecodeMessage.message_id
                        content = DecodeMessage.message_content

                        # locate room 
                        if messageid != 'clientdata':
                            print( f'\nAccepted sock {str(SOCK.getsockname())} recieved message {messageid}!' )
                            if room_id not in _rooms:
                                print(f"Detected room{room_id} first and create this room!")
                                _rooms[room_id] = Room(room_id)
                            else:
                                print(f"Detected exist room{room_id}!")
                        room = _rooms[room_id]

                        # handle different messageid
                        if(messageid=='clientroomenterrequest'):              
                            room.add_member(SOCK, content)
                    
                        if(messageid=='clientroommasterstart'):                        
                            if 'SSVEP_QC' == room.members_info[room.master]['paradigm']:
                                room.gen_startpara_for_SSVEPQC()

                        if(messageid=='clientdata'):
                            room.add_data(SOCK, content) 
                            continue

                        if(messageid=='clientdataover'):
                            room.add_dataover(SOCK, content) 
                            
                            value_len = [len(i) for i in room.members_marker.values()]
                            if len(set(value_len))==1:
                                room.process_data()

                                if room.room_times < room.max_room_times:
                                    room.gen_hintpara_for_SSVEPQC()
                                else:
                                    room.gen_lastpara_for_SSVEPQC()
                                
                                room.room_times += 1

                        if(messageid=='clientclosesocket'):
                            print(f"Client {str(SOCK.getpeername())} is normally offline!\n")
                            print(f'Clear this sock related info!')
                            room.del_member(SOCK)
                            if(len(room.members_info)==0):
                                print(f'Room{room_id} is deleted!\n')
                                del room
                                del _rooms[room_id]
                            _current_in_list.remove(SOCK)
                            SOCK.close()   
                            continue

                        # print new room info
                        print(f"Current Info of this room{room.roomID} after handle message id and message content:  ")
                        room.print_room()

                except socket.error:
                    if SOCK in _current_in_list:
                        print(f"\nClient {str(SOCK.getpeername())} is accidently offline!\n")
                        print(f'\nClear all rooms!\n')
                        while len(_rooms):
                            for room_id in _rooms.keys():
                                del _rooms[room_id]
                        _current_in_list.remove(SOCK)
                        SOCK.close()




