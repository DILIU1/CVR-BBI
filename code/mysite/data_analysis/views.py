from django.shortcuts import render

# Create your views here.
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import numpy as np
import pandas as pd

from server import EEG_data
from utils import *
import base64
import os
from django.http import HttpResponse
import zipfile
import tempfile
import json
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

logger = logging.getLogger(__name__)
import json

@csrf_exempt
def upload_data_matrix(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print("Received data:", data)  # 打印接收到的数据

            data_name = data.get('data_name', '')
            data_value = data.get('data_value', [])
            print("Data name:", data_name)  # 打印解析后的 data_name
            print("Data value:", data_value)  # 打印解析后的 data_value

            # 打印第一个数据
            if data_value and isinstance(data_value, list) and len(data_value) > 0:
                first_data = data_value[0]  # 假设 data_value 是一个一维数组
                print(f"First data: {first_data}")

                # 打印最后一个数据
                last_data = data_value[-1]  # 假设 data_value 是一个一维数组
                print(f"Last data: {last_data}")

                # 打印数据长度
                data_length = len(data_value)
                print(f"Data length: {data_length}")

            # 处理接收到的数据
            if data_name and isinstance(data_value, list) and all(isinstance(i, (int, float)) for i in data_value):
                # 假设要返回的结果
                result_string = f"Processed {data_name}"
                result_double = sum(data_value) / len(data_value) if data_value else 0.0
                data_value = np.array(data_value)  # 确保数据是 NumPy 数组
                most_likely_freq = process_EEG_psd(data_value.reshape((32, 1000)))

                print(f"message:received successfully")
                return JsonResponse({
                    'status': 'success',
                    'message': 'Data received successfully',
                    'result_string': result_string,
                    'result_double': most_likely_freq
                })
            else:
                return JsonResponse({'status': 'error', 'message': 'Invalid JSON data format'}, status=400)
        
        except json.JSONDecodeError:
            print("Invalid JSON format")  # 打印错误信息
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON format'}, status=400)
    else:
        return JsonResponse({'status': 'error', 'message': 'Only POST method allowed'}, status=405)


def image_to_base64(image_path):
    """将图片文件转换为Base64编码"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def send_images_as_zip(image_paths):
    """将多张图片打包为ZIP文件并发送"""
    # 在内存中创建一个ZIP文件
    response = HttpResponse(content_type='application/zip')
    zip_file = zipfile.ZipFile(response, 'w')
    for image_path in image_paths:
        if os.path.exists(image_path):
            zip_file.write(image_path, os.path.basename(image_path))
    zip_file.close()
    response['Content-Disposition'] = 'attachment; filename="result.zip"'
    return response

@csrf_exempt
@require_http_methods(["POST"])

def upload_signal(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file part'}, status=400)
        file = request.FILES['file']
        # 假设上传的是.npy文件
        try:
            # 创建一个临时文件
            temp_file_path = r'temp\temp.npy'
            
            request.session['uploaded_file_path'] = temp_file_path
            
            signal_array = np.load(file)
            print(os.getcwd())
            print(temp_file_path)
            np.save(temp_file_path,signal_array)
            # 在这里添加对signal_array的分析处理代码
            # 例如，计算其平均值并返回
            # 获取数组的维度信息，这里我们只返回第二维的长度
            chanel_num, length = signal_array.shape

            return JsonResponse({'length': length, 'chanel_num':chanel_num,'success': True})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500) 
        
@csrf_exempt
@require_http_methods(["POST"])       
def analyze_data(request):
  
    data = json.loads(request.body)
    start = data.get('start')
    end = data.get('end')
    
    if start is None or end is None:
        return JsonResponse({'error': 'Missing start or end value.'}, status=400)

    
    # 确保start和end是整数
    start = int(start)
    end = int(end)
    if start >= end :
        return JsonResponse({'error': 'Error value .'}, status=400)
    print(start,end)
    temp_file_path = r'temp\temp.npy'
    print(temp_file_path)
    if not temp_file_path or not os.path.exists(temp_file_path):
        return JsonResponse({'error': 'Uploaded file not found.'}, status=404) 
    # 使用保存的临时文件进行分析
    signal_array = np.load(temp_file_path, allow_pickle=True)
    
    image_paths=process_EEG_SSVEP(signal_array)

    os.remove(temp_file_path)
    del request.session['uploaded_file_path']

    return send_images_as_zip(image_paths)

def process_EEG_psd(final_array):
    chname_dict = {'EEG':['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'],
                    'MEG':['O01z', 'O02z', 'O03z', 'O05z', 'O06z', 'O07z', 'O11z', 'O12z', 'O13z', 'O14z', 'O15z', 'O16z', 'O17z', 'O18z', 'O19z', 'O110z', 'O111z', 'O21z', 'O22z', 'O23z', 'O24z', 'O25z', 'O26z', 'O27z', 'O28z', 'O29z', 'O31z', 'O32z', 'O33z']}
    ##data =  final_array[:,2000:5000]
    modality = 'EEG'

    _EEG_data = EEG_data(final_array,ch_names=chname_dict[modality],sfreq=1000,
                                    trialbaseline=200,minusbaseline=True,
                                    fmin=0.01,fmax=100,
                                    nf=[50,100,150],
                                    ch_bad=list(range(1,24))
                                    )

    (max1_freqs, max1_values, max2_freqs, max2_values), rhythm_powers ,(most_likely_freq, most_likely_amplitude)= _EEG_data.compute_psd_features(fmin=2, fmax=100, picks=['O1','Oz','O2','POz'])

    print("Max frequencies:", max1_freqs)
    print("Max frequencies values:", max1_values)
    print("Second max frequencies:", max2_freqs)
    print("Second max frequencies values:", max2_values)
    print("Rhythm powers:", rhythm_powers)
    print("Most likely frequency:", most_likely_freq)
    print("Most likely amplitude:", most_likely_amplitude)
    return most_likely_freq
     


def process_EEG_SSVEP(final_array):
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
        # 计算 PSD 特征
    fmin = 1  # 最小频率
    fmax = 50  # 最大频率
    nf = 50  # notch filter frequency

    _EEG_data.calc_psd(figfile='dataset/temp_result/test1.png',
                        fmin=fmin, fmax=fmax, nf=nf, picks=['O1','Oz','O2','POz']
                                )
    centers = _EEG_data.TimeFrequencyCWT(picks=_EEG_data.ch_names,
                                    figfile='dataset/temp_result/test2.png',
                                    waveinter=1,
                                    fmin=5,
                                    fmax=20
                                        )
    _EEG_data.temporal_plot(figfile='dataset/temp_result/test3.png',
                                median_plot=False,
                                legend=False,
                                locs=[200,400,600,800,1000])
    return ["dataset/temp_result/test1.png","dataset/temp_result/test2.png","dataset/temp_result/test3.png"]

def submit(request):
    return render(request, "submit/upload.html")