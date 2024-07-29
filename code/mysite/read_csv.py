import pandas as pd
import django
from django.utils.dateparse import parse_datetime
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")
django.setup()

from bciDB.models import TrialData



def preprocess_boolean(value):
    """将-1转换为False，1转换为True"""
    if value == -1:
        return False
    elif value == 1:
        return True
    return None  # 或其他适当的默认值

# 加载CSV数据
csv_file_path = r'D:\django\mysite\dataset\data_deindentification_dict.csv'  # 更新为你的CSV文件路径
data = pd.read_csv(csv_file_path)

# 预处理数据
# 假设 'boolean_field' 是需要转换的布尔字段
# 请根据实际情况替换 'boolean_field' 为你的字段名

# 如果需要，预处理布尔值字段
# 示例：data['scibackground'] = data['scibackground'].apply(preprocess_boolean)
# 根据你的PDF和实际字段应用上述转换

# 遍历DataFrame的每一行，创建并保存Django模型实例
for _, row in data.iterrows():
    # 解析日期字段，确保与你的模型字段匹配
    # 示例：date_parsed = parse_datetime(row['date']) if pd.notnull(row['date']) else None
    
    try:
        trial_data_instance = TrialData(
            # 假设模型字段与CSV列名完全一致
            address=row['address'],
            paradigm=row['paradigm'],
            modality=row['modality'],
            date=parse_datetime(row['date']) if pd.notnull(row['date']) else None,
            subject=row['subject'],
            label=row['label'],
            trialid=row['trialid'],
            trialsample=row['trialsample'],
            trialsfreq=row['trialsfreq'],
            trialchan=row['trialchan'],
            sexual=row['sexual'],
            age=row['age'],
            scibackground=(row['scibackground']),
            sleephours=row['sleephours'],
            sleepqaulity=row['sleepqaulity'],
            brainhealth=row['brainhealth'],
            workout=(row['workout']),
            exptime=row['exptime'],
            trialbaseline=row['trialbaseline'],
            # 确保你的模型字段名与CSV列名匹配
        )
        trial_data_instance.save()
    except Exception as e:
        print(f"Error saving row: {e}")

print("CSV数据导入完成。")