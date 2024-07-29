# 使用官方 Python 运行时作为父镜像
FROM python:3.8-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 设置工作目录
WORKDIR /app


# 复制项目依赖文件到容器中
COPY requirements.txt /app/

# 安装 Python 依赖
RUN pip install -r requirements.txt

# 复制项目文件到容器中
COPY . /app/

# 暴露端口
EXPOSE 8000

# 运行 Django 服务
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
