CVR-BBI
Welcome to the GitHub repository for the CVR-BBI project, a cutting-edge Collaborative Virtual Reality Brain-Brain Interface platform. This project integrates VR and BCI technologies to allow real-time brain-to-brain communication and interaction in a virtual environment.

Repository Structure
This repository contains several key components which are structured as follows:

Code
mysite: The Django server project, serving as the backbone of the application. This includes:
Global routes and configurations.
Database management system (bciDB).
WebSocket application (chat) for maintaining long connections.
Data analysis tools (data_analysis) for offline data processing.
Dataset handling and caching (dataset).
Release
This directory contains stable release versions of the software for deployment.
Documentation
README.md: Provides an overview of the project, installation instructions, and other necessary documentation to get started with the CVR-BBI.

Getting Started
Prerequisites
Python 3.8
Django 3.x
Other dependencies listed in requirements.txt
Installation

Clone the repository:
git clone https://github.com/username/CVR-BBI.git

Navigate to the project directory:
cd CVR-BBI

Install the required packages:
pip install -r requirements.txt

Running the Application
Navigate to the mysite directory:
cd mysite

Run the Django server:
python manage.py runserver

Docker Deployment
To deploy using Docker, follow these steps:

Build the Docker image:
docker build -t mysite .

Run the container:
docker run -d -p 8000:8000 mysite


Contributing
Contributions to the CVR-BBI project are welcome. Please refer to our contributing guidelines for more information on how to participate.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For any queries regarding the project, please contact di.liu@seu.edu.cn.