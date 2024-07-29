# chat/consumers.py
import json

from channels.generic.websocket import WebsocketConsumer

from channels.generic.websocket import AsyncWebsocketConsumer
import threading
from dataset import dataManage 
class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        # 启动心跳检测
        self.heartbeat()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json.get("message")

        # 检测到心跳响应
        if message == "pong":
            print("Heartbeat response received")
        else:
            self.send(text_data=json.dumps({"message": message}))
            
    def heartbeat(self):
        # 发送心跳（ping）消息
        self.send(text_data=json.dumps({"message": "ping"}))
        # 每30秒发送一次心跳
        threading.Timer(30, self.heartbeat).start()
        
import numpy as np
import pandas as pd

from server import EEG_data
from utils import *

class MyConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.array_list = []  # 用于存储接收到的数组
        self.is_collecting = False  # 标志是否开始收集数组
        await self.accept()

    async def disconnect(self, close_code):
        # 客户端断开连接时，如果有未处理的数据，则处理并保存
        if self.array_list:
            self.process_and_save_arrays()
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message_type = text_data_json['type']

        if message_type == 'start':
            # 开始收集数组
            self.array_list = []  # 重置数组列表
            self.is_collecting = True
        elif message_type == 'data' and self.is_collecting:
            # 收集数据
            array = np.array(text_data_json['array'])
            self.array_list.append(array)
            print(f"Received array with shape: {array.shape}")
        elif message_type == 'end':
            # 结束收集并处理数据
            self.is_collecting = False
            self.process_and_save_arrays()
            self.array_list = []

    def process_and_save_arrays(self):
        # 拼接所有数组并进行处理
        if self.array_list:
            final_array = np.concatenate(self.array_list, axis=1)
            # 在这里添加对final_array的分析处理步骤
            print(f"Final array shape: {final_array.shape}")
            # 保存处理后的数组
            self.process_EEG_SSVEP(final_array)
            csv_file_path = r'/user/diliu/mysite/dataset/data_deindentification_dict.csv'
            dataManage.add_trial_data_to_csv(csv_file_path,final_array)
            print("Array processing completed and saved.")
            
    def process_EEG_SSVEP(self,final_array):
        chname_dict = {'EEG':['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'],
'MEG':['O01z', 'O02z', 'O03z', 'O05z', 'O06z', 'O07z', 'O11z', 'O12z', 'O13z', 'O14z', 'O15z', 'O16z', 'O17z', 'O18z', 'O19z', 'O110z', 'O111z', 'O21z', 'O22z', 'O23z', 'O24z', 'O25z', 'O26z', 'O27z', 'O28z', 'O29z', 'O31z', 'O32z', 'O33z']}
        data =  final_array[:,2000:5000]
        modality = 'EEG'

        _EEG_data = EEG_data(data,ch_names=chname_dict[modality],sfreq=1000,
                                        trialbaseline=200,minusbaseline=True,
                                        fmin=0.01,fmax=100,
                                        nf=[50,100,150],
                                        ch_bad=list(range(1,24))
                                        )
        _EEG_data.temporal_plot(figfile='dataset/temp_result/test1.png',
                                    median_plot=False,
                                    legend=False,
                                    locs=[200,400,600,800,1000]
                                    )
        centers = _EEG_data.TimeFrequencyCWT(picks=_EEG_data.ch_names[-9:],
                                        figfile='dataset/temp_result/test2.png',
                                        waveinter=1,
                                        fmin=5,
                                        fmax=20
                                            )
        
