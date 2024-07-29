
import pandas as pd
from datetime import datetime
import numpy as np
def add_trial_data_to_csv(csv_file_path, npy_data,address="nanjing_seuallen", paradigm="SSVEP_QC", modality="EEG",
                          date=None, subject="test", label="10",
                          trialid="trial_1", trialsample=5000, trialsfreq=1000,
                          trialchan="['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2', 'AFz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4']",
                          sexual="m", age=25, scibackground=1, sleephours=0.5, sleepqaulity=-1.0,
                          brainhealth="health", workout=-1.0, exptime=0.5, trialbaseline=0):
    # Load the existing data from the CSV file
    df = pd.read_csv(csv_file_path)
    if  not date:
        date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(date)
    # 计算新的"Unnamed"序号值
    if "Unnamed: 0" in df.columns:
        new_unnamed_value = df["Unnamed: 0"].max() + 1
    else:
        new_unnamed_value = 0
    # Create a new row with the given parameters
    new_row = pd.DataFrame([{
        "Unnamed: 0": new_unnamed_value,
        "address": address, "paradigm": paradigm, "modality": modality, "date": date,
        "subject": subject, "label": label, "trialid": trialid, "trialsample": trialsample,
        "trialsfreq": trialsfreq, "trialchan": trialchan, "sexual": sexual, "age": age,
        "scibackground": scibackground, "sleephours": sleephours, "sleepqaulity": sleepqaulity,
        "brainhealth": brainhealth, "workout": workout, "exptime": exptime, "trialbaseline": trialbaseline
    }])
    
    # 使用pandas.concat来添加新行
    df_updated = pd.concat([df, new_row], ignore_index=True)
    
    # Save the updated dataframe back to the CSV file
    df_updated.to_csv(csv_file_path, index=False)
    np.save('D:/django/mysite/dataset/'+str(int(new_unnamed_value))+'.npy', npy_data)
    return "Data added and CSV file updated."

# Let's apply the function to add a sample row to our dataframe and display the last few rows to verify


# csv_file_path = r'D:\django\mysite\dataset\data_deindentification_dict.csv'
# Load the CSV file
# data = pd.read_csv(csv_file_path)

# # Get the number of records
# print(len(data))
# result = add_trial_data_to_csv(csv_file_path)
# print(result)
# # Load the CSV file
# data = pd.read_csv(csv_file_path)

# # Get the number of records
# print(len(data))
