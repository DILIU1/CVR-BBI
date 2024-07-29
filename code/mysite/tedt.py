import asyncio
import websockets
import numpy as np
import json

async def send_arrays():
    uri = "ws://10.192.34.48:8000/ws/senddata/"
    async with websockets.connect(uri) as websocket:
        # 发送开始信号
        await websocket.send(json.dumps({'type': 'start'}))
        
        # 计算发送次数，因为每2秒发送一次，所以15秒内发送7次
        for _ in range(7):
            # 创建一个（32, 2000）的随机浮点数数组
            array = np.random.rand(32, 2000).tolist()  # 转换为列表以便序列化
            await websocket.send(json.dumps({'type': 'data', 'array': array}))
            await asyncio.sleep(2)
        
        # 发送结束信号
        await websocket.send(json.dumps({'type': 'end'}))
        
        # 给服务器一点时间来处理结束信号
        await asyncio.sleep(1)

asyncio.run(send_arrays())
