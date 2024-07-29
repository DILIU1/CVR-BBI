function uploadFile() {
    const input = document.getElementById('fileInput');
    const file = input.files[0];
    const uploadButton = document.getElementById('uploadButton');
    const feedbackElement = document.getElementById('feedback');
    if (!file) {
        alert("Please select aaaa file.");
        return;
    }
    // 禁用上传按钮并显示加载提示
    uploadButton.disabled = true;
    feedbackElement.textContent = "Uploading...";

    // 使用FormData对象封装文件数据
    const formData = new FormData();
    formData.append('file', file);

    // 发送POST请求到服务器
    fetch('http://127.0.0.1:8000/data_analysis/upload/', {  // 替换为你的实际上传接口URL  http://114.117.165.134:8512
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.success && 'length' in data) {
            setupSliders(data.length,data.chanel_num); // 根据返回的长度设置滑块
            document.getElementById('sliderContainer').style.display = 'block'; // 显示滑块
            updateSliderValues() ;
        }
    })
    .catch(error => {
        console.error('Error fetching data:', error);
    });
    uploadButton.disabled = false;
}


function analyzeData() {
    const startValue = document.getElementById('rangeStart').value;
    const endValue = document.getElementById('rangeEnd').value;
    const feedbackElement = document.getElementById('feedback');
    const analyzeButton = document.getElementById('analyzeButton');
    analyzeButton.disabled = true;
    // 发送POST请求到服务器
    fetch('http://127.0.0.1:8000/data_analysis/analysis/', {  // 替换为你的实际上传接口URL
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ start: startValue, end: endValue }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.blob();
    })
    .then(blob => {
        // 创建下载链接
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = "analysis_result.zip"; // 假设服务器返回的是ZIP文件
        link.textContent = "Download Analysis Result";
        
        const downloadDiv = document.getElementById('downloadLink');
        downloadDiv.appendChild(link);
        feedbackElement.innerHTML = '';
        feedbackElement.appendChild(link);
        feedbackElement.appendChild(document.createTextNode(" Upload successful."));
    })
    .catch(error => {
        // 处理错误，显示错误消息
        console.error('Error:', error);
        feedbackElement.textContent = "Error: " + error.message;
    })
    .finally(() => {
        // 无论成功还是失败，恢复按钮的可用状态
        analyzeButton.disabled = false;
    });
}


// 假设后端返回的数据长度已经保存在变量dataLength中
function setupSliders(dataLength,chanel_num) {
    const rangeStart = document.getElementById('rangeStart');
    const rangeEnd = document.getElementById('rangeEnd');
    const startValue = document.getElementById('startValue');
    const endValue = document.getElementById('endValue');
    const Chanel = document.getElementById('Chanel');
    // 设置滑块的最大值为数据长度
    rangeStart.max = dataLength;
    rangeEnd.max = dataLength;
    rangeEnd.value = dataLength; // 默认终点为数据长度
    Chanel.textContent = chanel_num;
    // 更新显示的范围值
    startValue.textContent = rangeStart.value;
    endValue.textContent = rangeEnd.value;
}

// 用户滑动滑块时更新显示的值
function updateSliderValues() {
    const startValue = document.getElementById('startValue');
    const endValue = document.getElementById('endValue');
    const analyzeButton = document.getElementById('analyzeButton');
    startValue.textContent = rangeStart.value;
    endValue.textContent = rangeEnd.value;
    // 根据起点和终点的值启用或禁用分析按钮
    if(parseInt(rangeStart.value) < parseInt(rangeEnd.value))
        analyzeButton.disabled = false;
    else
        analyzeButton.disabled = true;
}
