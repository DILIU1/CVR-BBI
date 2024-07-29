# data identification dict
'SSVEP_QC' data saved by yingxinli
'P300_shape' data saved by yingxinli
'Emotion_DEAP' data saved by changshanli
# explaination of each column
address: exp spot
paradigm: as we designed in ui
          if paradigm=='SSVEP_QC', label is in [10,13]
          if paradigm=='P300_shape', label is in ['triangle_background',...]
          if paradigm=='Emotion_DEAP', label is ['emotion1','emotion2','emotion3','likety']
modality: eeg/meg
date: exp time, -1 if unknown
subject: subject name, if unknown, will use random code instead
label: the true label of each trial
trialid: the trial id in each sequence
trialsample: the length of the acquired data
trialsfreq: the sampling rate of acquiring data
trialchan: the data montage
trialbaseline: the data length for the usage as baseline
sexual: 'm'/'f' for man/female
age: -1 if unknown
scibackground: 	whether the subject has neuroscience background: 1 for yes, 0 for no, -1 for unknown
sleephours: whether the subject sleep enough before exp: 1 for yes, 0 for no, -1 for unknown
sleepqaulity: whether the subject sleep tight before exp: 1 for yes, 0 for no, -1 for unknown
workout: whether the subject work out often, 0 for no, 0.5 for sometimes, 1 for often, -1 for unknown
exptime: 0 for morning, 0.5 for afternoon, 1 for evening, -1 for unknown	
brainhealth: the brain state of the subject, 'health', 'damaged', 'epilepsy'...
# if use python to read csv, please note that 
pd.read_csv() must have 'index_col=1' !
# note that appended on 20240110: 
some data has problem, does not match log, delete all related sliced data in backup20231220, therefore deleted all related rows in dict, and deleted deindentification data


# EVR
BioElectric driven VR application development

# server-client mode
1. server.py should be capable of being planted onto different machine with only sc.py, utils.py, montage.csv
2. for every client, there are client_config.py, sc.py, utils.py, *.ui, *.ui.py, /res/* to setup

# server.py
1. keep main socket
2. once accept new client coonection, handle and remember room id, and delete socket

# client.py
1. every client load local client_config and react different dialog with the EXP.PARADIGM
2. if SSVEP_QC, new SSVEPQCDialog, for this UI
3. once 'connect' is click, produce a new thread to keep listening Server
4. once 'start' is click, message send to server
5. once message recieved, handle
6. once message is 3 or 201, decoded EEG, NOTE here test_send_result is for test, should be replaced by real EEG docoding message

# define EEG format and function
1. only in server.py

# Support different ui window in client.py
# NOTE: we should develop home page and slots for principles to choose different paradiam and open related ui
1. If 'SSVEP_QC.ui' were selected, class Ui_MainWindow will be imported from SSVEP_QC.ui.py
2. If 'P300_shape.ui' were selected, class Ui_MainWindow will be imported from P300_shape.ui.py

# Current Version Usage
1. Write config.py to set global variables