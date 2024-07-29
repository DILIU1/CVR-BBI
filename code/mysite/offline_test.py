import os
import re
import math
import numpy as np
import pandas as pd

from server import EEG_data
from utils import *



# bh ant
# datafile = r'C:\Users\user\Desktop\eegdata\default2023-11-01-11-27-25total.npy'
# fix = datafile.split("\\")[-1]
# _EEG_data = EEG_data(data = datafile,
#                      ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'IO', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10', 
#                     'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Fpz', 'CPz', 'POz', 'Oz'],
#                      ch_nums = 32,
#                      sfreq = 1000
#                      )
# _EEG_data.calc_psd(picks=['O1','Oz','O2'],figfile=f'{fix}_freq.png')



# # test psd
# datadir = r'G:\BCIteam_Allrelated\SharedSource\bciBASE\server_data'
# df = pd.read_csv(r'G:\BCIteam_Allrelated\SharedSource\bciBASE\data_deindentification_dict.csv',index_col=0)
# ii = 5524
# for i,row in df.iterrows():
#     if i==ii:
#         datafile = rf'{datadir}\{row["paradigm"]}\{row["modality"]}\{row["label"]}\{i}.npy'
#         ch_names = eval(row["trialchan"])
#         ch_nums = len(ch_names)
#         _EEG_data = EEG_data(datafile,ch_names=ch_names)
#         _EEG_data.calc_psd(picks=_EEG_data.ch_names,figfile='test.png',fmin=0,fmax=60)   



# # fast ica test
# datadir = r'G:\BCIteam_Allrelated\SharedSource\bciBASE\server_data'
# df = pd.read_csv(r'G:\BCIteam_Allrelated\SharedSource\bciBASE\data_deindentification_dict.csv',index_col=0)
# ii = 5524
# for i,row in df.iterrows():
#     if i==ii:
#         datafile = rf'{datadir}\{row["paradigm"]}\{row["modality"]}\{row["label"]}\{i}.npy'
#         ch_names = eval(row["trialchan"])
#         ch_nums = len(ch_names)
#         _EEG_data = EEG_data(datafile,ch_names=ch_names,ch_nums=ch_nums)
#         Sr, loc = _EEG_data.calc_topo(para=None,return_loc=True)
#         print(loc)
        
#         w = math.ceil(math.sqrt(ch_nums+1))
#         h = math.ceil((ch_nums+1)/w)
#         fig = plt.figure(figsize=(w*3,h*3))
#         iax = 1
#         ax = fig.add_subplot(w,h,iax)
#         _EEG_data.plot_topo(para=np.zeros((ch_nums)),ax=ax)
#         ax.set_title('eeg record chans')
#         for ic in range(Sr.shape[0]):
#             para=Sr[ic,:]
#             # _EEG_data.plot_topo(para = para,figfile=rf'.\sourceweight_chan{ic+1}{_EEG_data.ch_names[ic]}_data{i}.png',ax=None,loc=loc)
#             ax = fig.add_subplot(w,h,ic+2)
#             _EEG_data.plot_topo(para=para,loc=loc,ax=ax)
#             ax.set_title(rf'for {ch_names[ic]}')
#         figfile=rf'.\sourceweight_foreverychan_data{i}.png'
#         plt.savefig(figfile)
#         plt.close()



# # contourf test
# datadir = r'G:\BCIteam_Allrelated\SharedSource\bciBASE'
# df = pd.read_csv(rf'{datadir}\data_deindentification_dict.csv',index_col=0)
# ii = 5524
# for i,row in df.iterrows():
#     if i==ii:
#         datafile = rf'{datadir}\server_data\{row["paradigm"]}\{row["modality"]}\{row["label"]}\{i}.npy'
#         data = np.load(datafile)
#         data = data.mean(axis=1)

#         # normalization
#         # data = (data - data.min()) / (data.max() - data.min()) # old way
#         scale = np.nanmax(abs(data))
#         if scale!=scale: 
#             scale=np.nan
#         data = data / scale

#         ch_names = eval(row["trialchan"])
#         ch_nums = len(ch_names)
#         pd20 = pd.read_csv(f'{datadir}/montage.csv',index_col=0)
#         pd32 = pd20.loc[ch_names] # ,x='RIGHT',y='NOSE'

#         x = pd32['RIGHT'].sort_values()
#         y = pd32['NOSE'].sort_values()
#         x = ( x/(x.values[1:]-x.values[:-1]).min() ).round().apply(int)
#         x = x - x.min()
#         y = ( y/(y.values[1:]-y.values[:-1]).min() ).round().apply(int)
#         y = y - y.min()
#         z = np.zeros( (y.max()+1, x.max()+1) )
#         z[z==0]=np.nan
#         for ich, ch in enumerate(pd32.index):
#             z[ y.loc[ch] , x.loc[ch] ] = data[ich]

#         fig = plt.figure(figsize=(2,2))
#         ax = fig.add_subplot(1,1,1)
#         print(z.shape)
#         ax.contourf(z)
#         ax.set_title('contour')

#         figfile=rf'.\test.png'
#         plt.savefig(figfile)
#         plt.close()

#         # import pdb;pdb.set_trace()       
#         # sx = x[int(len(x)/2)]
#         # sy = y[int(len(y)/2)]
#         # shift = 0 
#         # while z.sum()==np.nan:
#         #     shift += 1
#         #     for y in range(sy-shift,sy+shift):
#         #         for x in range(sx-shift,sx+shift):
#         #             if y in range(z.shape[0]) and x in range(z.shape[1]):
#         #                 pas
#         #sns.scatterplot(pd32,c='black',ax=ax)



# test vr
# csvdir = rf'G:\BCIteam_Allrelated\SharedSource\bciVR\rawData\2024-1-30-SSVEP-VR'
# filelist = [ i for i in os.listdir(csvdir) if i.endswith('.csv')]
# for filename in filelist:
#     csvfile = rf'{csvdir}\{filename}'
#     print(csvfile)
#     csv = pd.read_csv(csvfile,header=None)
#     data = csv.values[:32,20000:30000]
#     print(data.shape)

#     _EEG_data = EEG_data(data)
#     _EEG_data.calc_psd(picks=['Fp1','P3','Pz','P4','O1','Oz','O2','POz'],
#                        figfile=rf'{csvfile}_psd.png',
#                        snr=True,
#                        snrfile=rf'{csvfile}_snr.png')
#     _EEG_data.plot_TimeFrequencyCWT(picks=['Fp1','P3','Pz','P4','O1','Oz','O2','POz'],
#                                     figfile=rf'{csvfile}_cwt.png',
#                                     fmin=8,
#                                     fmax=16)

# datafile = rf'C:\Users\user\Documents\WeChat Files\wxid_cphpiiex21mv22\FileStorage\File\2024-02\data_after.npy'
# datafile = rf'C:\Users\user\Documents\WeChat Files\wxid_cphpiiex21mv22\FileStorage\File\2024-02\SSVEP2024-02-25-16-24-27_13hz.csv'
# _EEG_data = EEG_data(datafile)
# _EEG_data.plot_TimeFrequencyCWT(picks=['O1','Oz','O2','POz'],
#                                     figfile=rf'test.png',
#                                     fmin=5,
#                                     fmax=20,
#                                     nf=50)


# all func test
# for i, datafile in enumerate([
#                  rf'G:\BCIteam_Allrelated\SharedSource\bciBASE\server_data\SSVEP_QC\MEG\10\3763.npy',
#                  rf'G:\BCIteam_Allrelated\SharedSource\bciBASE\server_data\SSVEP_QC\EEG\10\6.npy',
                 
#                  rf'G:\BCIteam_Allrelated\SharedSource\bciVR\rawData\2024-1-30-SSVEP-VR\SSVEP2024-01-30-15-37-38_10hz.csv',
#                  rf'G:\BCIteam_Allrelated\SharedSource\bciBASE\server_data\P300_shape\EEG\square_background\9532.npy',
#                  rf'D:\WXWork\1688850447499607\Cache\File\2023-12\1222_10.vhdr',
#                  ]):
#     print('\n\n\n',i,datafile) 
#     if 'MEG' in datafile:
#         _EEG_data = EEG_data(datafile,ch_names=['O01z', 'O02z', 'O03z', 'O05z', 'O06z', 'O07z', 'O11z', 'O12z', 'O13z', 'O14z', 'O15z', 'O16z', 'O17z', 'O18z', 'O19z', 'O110z', 'O111z', 'O21z', 'O22z', 'O23z', 'O24z', 'O25z', 'O26z', 'O27z', 'O28z', 'O29z', 'O31z', 'O32z', 'O33z'])
#     else:
#         _EEG_data = EEG_data(datafile)
#     fmin = 0.01
#     fmax = 100
#     nf = [50,100,150]    
    # _EEG_data.temporal_plot(para=None,
    #                         figfile=rf'{i}_{fmin}_{fmax}_{nf}_raw.png',
    #                         median_plot=True,
    #                         legend=True,locs=[1,500,1000,1500],
    #                         fmin=fmin,fmax=fmax,nf=nf)
    # _ = _EEG_data.calc_psd(fmin=fmin,fmax=fmax,nf=nf,
    #                    picks=_EEG_data.ch_names[-10:], 
    #                    figfile=rf'{i}_{fmin}_{fmax}_{nf}_psd.png',
    #                    snr=True,snrfile=rf'{i}_{fmin}_{fmax}_{nf}_snr.png')
    # print(_)
    # _ = _EEG_data.calc_psd(fmin=fmin,fmax=fmax,nf=nf,
    #                    picks=_EEG_data.ch_names[-10:], 
    #                    clac_on_rhythms=True)
    # print(_)
    # _ = _EEG_data.calc_features(feature_names=['len','freq','amplitude','latency_max','latency_min','latency_extremum',
    #                                            'std','skew','kurtosis','de10','zcr','olr',
    #                                            'rhythm'
    #                                            ],                                          
    #                             fmin=fmin,fmax=fmax,nf=nf,
    #                             # calc_median=False, # test that all feature not median-related
    #                             figfile=rf'{i}_{fmin}_{fmax}_{nf}_feature.png'
    #                             )
    # _.to_csv('test.csv')
    # # test center return
    # _ = _EEG_data.plot_TimeFrequencyCWT(picks=_EEG_data.ch_names[-5:],
    #                     return_center=False, # note to test latter
    #                     figfile=rf'{i}_{fmin}_{fmax}_{nf}_cwt.png',
    #                     fmin=5,fmax=20,nf=[50,100,150]) # test the 2-f notcher to be available
    # print(_)
    # break



# position test
# ch_names=['O01z', 'O02z', 'O03z', 'O05z', 'O06z', 'O07z', 'O11z', 'O12z', 'O13z', 'O14z', 'O15z', 'O16z', 'O17z', 'O18z', 'O19z', 'O110z', 'O111z', 'O21z', 'O22z', 'O23z', 'O24z', 'O25z', 'O26z', 'O27z', 'O28z', 'O29z', 'O31z', 'O32z', 'O33z']
# ch_names=['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz',
#              'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2']
# _EEG_data = EEG_data(data=np.ones((len(ch_names),1000))*np.nan,ch_names=ch_names)
# _EEG_data.plot_topo(para=0,figfile='test.png',sphere=0.11,sensors='number')

# pd20 = pd.read_csv(f'./montage.csv',index_col=0)
# pd32 = pd20.loc[ch_names]
# # print(pd32.max(axis=0))
# # print(pd32.min(axis=0))

# # todel
# data = [0.3,5,-0.7,0.4]*8
# for i,d in zip(ch_names,data):
#     print(i,d)

# mne.viz.plot_topomap(data[:len(ch_names)],
#                     pos = pd32.values[:,:2],
#                     sensors=True,
#                     names = pd32.index,
#                     contours=10,
#                     outlines='head',
#                     sphere=0.12, 
#                     extrapolate='local',
#                     border='mean',
#                     cmap = 'jet', 
#                     show = True)

# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(111,projection='3d')
# for i,row in pd32.iterrows():
#     xx,yy,zz = row.values
#     ax.scatter(xx,yy,zz,c='r',s=3)
#     ax.text(xx, yy, zz, i)
# ax.set_xlim([-0.1,0.1])
# ax.set_ylim([-0.1,0.1])
# ax.set_zlim([-0.1,0.1])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()



# # test AMPD
# import random
# data = np.array([random.randint(1,100) for _ in range(1000)])
# x_peak = find_mypeaks(data,fp=False,figfile='test.png',ax=None,method='AMPD')
# print(x_peak)



# test filter
# chname_dict = {'EEG':['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'],
# 'MEG':['O01z', 'O02z', 'O03z', 'O05z', 'O06z', 'O07z', 'O11z', 'O12z', 'O13z', 'O14z', 'O15z', 'O16z', 'O17z', 'O18z', 'O19z', 'O110z', 'O111z', 'O21z', 'O22z', 'O23z', 'O24z', 'O25z', 'O26z', 'O27z', 'O28z', 'O29z', 'O31z', 'O32z', 'O33z']}
# modality = 'EEG'
# datafile = rf'H:\BCIteam_Allrelated\SharedSource\bciBASE\server_data\P300_shape\EEG\circle_background\9530.npy'
# # datafile = rf'H:\BCIteam_Allrelated\SharedSource\MEG_EEG\scripts\EEG_square_red_average.npy'
# # modality = 'MEG'
# # datafile = rf'H:\BCIteam_Allrelated\SharedSource\bciBASE\server_data\P300_shape\MEG\circle_red\12040.npy'
# data = np.load(datafile)
# length = data.shape[1]

# median = np.nanpercentile(data,0.5,axis=1)
# datamedian = np.repeat(np.array([median]),length,axis=0).T
# print(datamedian.shape)
# data = data - datamedian
# datapad = 0 * datamedian
# print(datapad.shape)
# data = np.hstack((datapad,data))
# print(data.shape)

# value1 = 48
# value2 = 52
# order = 8
# wn1 = 2*value1/1000 
# wn2 = 2*value2/1000 
# b, a  =   signal.butter( order ,  [wn1,wn2] ,  'bandstop' )    #配置滤波器 8 表示滤波器的阶数
# filtedData  =   signal.filtfilt(b, a, data) 
# filtedData = filtedData[:,length+1000:]

# _EEG_data = EEG_data(filtedData,ch_names=chname_dict[modality],sfreq=1000,
#                                  trialbaseline=data.shape[1],minusbaseline=False,
#                                  )
# _EEG_data.temporal_plot(figfile='test1.png',
#                             median_plot=False,
#                             legend=False,
#                             legend_cols=9999,
#                             locs=[0,200,400,600,800,1000]
#                             )
# _EEG_data.plot_TimeFrequencyCWT(picks=_EEG_data.ch_names,
#                                    figfile='test2.png',
#                                    returns = []
#                                     )



# # test fft
# datafile = rf'H:\BCIteam_Allrelated\SharedSource\bciBASE\server_data\P300_shape\EEG\circle_background\9530.npy'
# chname_dict = {'EEG':['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'],
# 'MEG':['O01z', 'O02z', 'O03z', 'O05z', 'O06z', 'O07z', 'O11z', 'O12z', 'O13z', 'O14z', 'O15z', 'O16z', 'O17z', 'O18z', 'O19z', 'O110z', 'O111z', 'O21z', 'O22z', 'O23z', 'O24z', 'O25z', 'O26z', 'O27z', 'O28z', 'O29z', 'O31z', 'O32z', 'O33z']}
# modality = 'EEG'
# data = np.load(datafile)
# _EEG_data = EEG_data(data,ch_names=chname_dict[modality],sfreq=1000,
#                                  trialbaseline=200,minusbaseline=True,
#                                  fmin=0.1,fmax=100,
#                                 #  nf=[50,100,150],
#                                  )
# _EEG_data.temporal_plot(figfile='test1.png',
#                             median_plot=False,
#                             legend=False,
#                             legend_cols=9999,
#                             locs=[0,200,400,600,800,1000]
#                             )
# _EEG_data.plot_TimeFrequencyCWT(picks=_EEG_data.ch_names,
#                                    figfile='test2.png',
#                                    returns = []
#                                     )
# new_data = data.copy()
# rm_freq = 50
# for d in range(new_data.shape[0]):
#     spec = np.fft.fft(new_data[d,:])
#     rm_ind = int(50*len(spec)/1000)
#     rm_pad = int(len(spec)/100)
#     print(rm_ind,rm_pad)
#     spec[rm_ind-rm_pad:rm_ind+rm_pad] = 0
#     recon = np.fft.ifft(spec)
#     new_data[d,:] = recon
# _EEG_data = EEG_data(new_data,ch_names=chname_dict[modality],sfreq=1000,
#                                  trialbaseline=200,minusbaseline=True,
#                                  fmin=0.1,fmax=100,
#                                 #  nf=[50,100,150],
#                                  )
# _EEG_data.temporal_plot(figfile='test3.png',
#                             median_plot=False,
#                             legend=False,
#                             legend_cols=9999,
#                             locs=[0,200,400,600,800,1000]
#                             )
# _EEG_data.plot_TimeFrequencyCWT(picks=_EEG_data.ch_names,
#                                    figfile='test4.png',
#                                    returns = []
#                                     )



# # # test projection

# # text1 = [1,2,3,4,2,4,5,5]
# # text2 = [1,2,6,2,4,-1,5,1,5,5]
# # res = longestCommonSubsequence(text1, text2)
# # print(res)

# chname_dict = {'EEG':['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'],
# 'MEG':['O01z', 'O02z', 'O03z', 'O05z', 'O06z', 'O07z', 'O11z', 'O12z', 'O13z', 'O14z', 'O15z', 'O16z', 'O17z', 'O18z', 'O19z', 'O110z', 'O111z', 'O21z', 'O22z', 'O23z', 'O24z', 'O25z', 'O26z', 'O27z', 'O28z', 'O29z', 'O31z', 'O32z', 'O33z']}

# datafile = rf'H:\BCIteam_Allrelated\SharedSource\bciBASE\server_data\P300_shape\EEG\circle_background\9530.npy'
# # # datafile = rf'H:\BCIteam_Allrelated\SharedSource\MEG_EEG\scripts\EEG_square_red_average.npy'
# # # datafile = rf'H:\BCIteam_Allrelated\SharedSource\bciBASE\server_data\P300_shape\MEG\circle_red\12040.npy'
# modality = 'EEG'

# _EEG_data = EEG_data(datafile,ch_names=chname_dict[modality],sfreq=1000,
#                                  trialbaseline=200,minusbaseline=True,
#                                  fmin=0.01,fmax=100,
#                                  nf=[50,100,150],
#                                  )
# _EEG_data.temporal_plot(figfile='test1.png',
#                             median_plot=False,
#                             legend=False,
#                             legend_cols=9999,
#                             locs=[0,200,400,600,800,1000]
#                             )
# centers = _EEG_data.TimeFrequencyCWT(picks=_EEG_data.ch_names,
#                                    figfile='test2.png',
#                                    returns = ['center'],
#                                     )

# # methods = 'fLCS' #'plv' #'tLCS'
# # values = _EEG_data.calc_projection(methods=methods,
# #                           mparas=None,
# #                           calc_median=False,
# #                           remove_baseline=True,
# #                           figfile=None
# #                             )
# # print(values)



# data test
chname_dict = {'EEG':['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'],
'MEG':['O01z', 'O02z', 'O03z', 'O05z', 'O06z', 'O07z', 'O11z', 'O12z', 'O13z', 'O14z', 'O15z', 'O16z', 'O17z', 'O18z', 'O19z', 'O110z', 'O111z', 'O21z', 'O22z', 'O23z', 'O24z', 'O25z', 'O26z', 'O27z', 'O28z', 'O29z', 'O31z', 'O32z', 'O33z']}

# datafile = rf'F:\git_all\BCIVR2.0\StreamingAssets\EEGDATA\2024-3-21-SSVEP-VR\SSVEP_old2024-03-21-10-04-21_7hz.csv'
# df = pd.read_csv(datafile,header=None)
data =  np.random.rand(32, 5000)
modality = 'EEG'

_EEG_data = EEG_data(data,ch_names=chname_dict[modality],sfreq=1000,
                                 trialbaseline=200,minusbaseline=True,
                                 fmin=0.01,fmax=100,
                                 nf=[50,100,150],
                                 ch_bad=list(range(1,24))
                                 )
_EEG_data.temporal_plot(figfile='test1.png',
                            median_plot=False,
                            legend=False,
                            locs=[200,400,600,800,1000]
                            )
centers = _EEG_data.TimeFrequencyCWT(picks=_EEG_data.ch_names[-9:],
                                   figfile='test2.png',
                                   waveinter=1,
                                   fmin=5,
                                   fmax=20
                                    )


