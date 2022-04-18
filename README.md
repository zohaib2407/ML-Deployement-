# ML-Deployement


I made a cloud API with a Restnet18 image classifier that works on image classes of CIFAR-10. The Resnet18 model was trained locally and the trained parameters are then saved. The model is called and loaded each time a POST request is made using the Heroku App.

## Introduction
MLOps follows a set of practices to deploy and maintain machine learning models in production efficiently and reliably. According to the Algorithmia report, nearly 22 percent of companies have had ML models in production for one to two years. Independent Research Firm Corinium Intelligence conducted the first-ever State of ModelOps & MLOps, surveying 100 AI-Focused Executives from Top Global financial services companies in North America and Europe.

MLOps is the amalgamation of machine learning, data engineering and DevOps. Through this project, we will explore different tools and technologies that assists Data Science professionals in bringing their machine learning models to production.

## Objective 

In this project,

•	We will build an image classifier using convolution neural network and Pytorch module from python. 

•	We will then save the model parameters and load the trained image classifier every time when predicting  new images using different tools and technologies from the MLOps community

•	We will deploy the image classifier on different remote platforms and make them accessible over internet with HTTP endpoints. We will use Python and Flask web framework with a flask app

•	We will use 4 different ways to demonstrate how we can deploy a deep neural network  model and make it respond to the HTTP requests with the predicted label of the image and the Image id

•	First, we will deploy the model on our remote virtual machine on AWS and access it from our local machine using HTTP via python’s inbuilt ‘request’ package

•	Second, we will build a publicly hosted web application using Heroku integrating the GitHub repository which will enable our application to run, take inputs from the user using HTML and CSS and display the predicted label and id

•	Third, we will use GCP Cloud run to containerize a Docker Image and deploy it. We will then make predictions using HTTP triggers to the docker container

•	Fourth, we will deploy our model using GCP Cloud Function and access it remotely to spill out predictions for new input images 

•	At the end we will draw a comparative evaluation of the different deployments used in the project

## Possible Solutions
Nowadays, there are wide variety of tools and technologies available for operationalizing machine learning model. There are both, open-source and paid services available for the same purpose. Few of the paid services and Amazon Web Services, Google Cloud, etc. and some freely available solutions are Heroku, Remote VM etc. However, AWS and GCP does provide a limited version of their services at zero costs. We have explored both the kinds in our project.  

## Model Building and Storing

We finetuned a convolution neural network based on Resnet-18 architecture for image classification that is available through Pytorch module. The classifier was trained on the CIFAR-10 dataset. The training dataset contains images from 10 different classes: airplane, automobile, car, cat, dog, deer, horse, frog, ship and truck. There were 5000 training images of each class, so our training dataset had a total of 50,000 images. After the model was trained, we saved the model parameters as a python object and the image labels as a pickle object.   


## Building the Flask Application and loading the trained classifier
Flask is a micro web framework written in python. We built a flask app using the function decorators from flask that helps to route our http requests to defined functions. The name of the routing function is ‘predict’ with ‘/’ and trigger. Next, once our model is deployed, we use ‘request’ module from python to send a ‘POST’ request along with an image to the flask application. Every time the routing function is triggered by our request, the incoming image is first pre-processed, loaded and converted into a tensor. Next, a new Resnet-18 neural network architecture is loaded from the ‘torch’ module, and then the model parameters we save after training are uploaded from local and then assigned to the neural network. This model is not provided the image tensor which was received using ‘POST’ request on which the prediction is made. Based on the ID that the model predicted, the label is picked from the pickle object that was saved earlier. In the end, the flask sends the predicted id and the label of the image to the client in a json format.

## 1.	Deploying the Pytorch model on Remote machine

In the first method, we will deploy and run our Pytorch based classifier on a remote machine. For this purpose, we leveraged AWS. We signed-up with AWS and opted for a free tier. Then we used AWS EC2 to get ourselves a remote machine. We rented a free-tier Ubuntu virtual machine on a remote server sitting somewhere at amazon’s data center. Once our remote machine was assigned and running, we used a SSH client, and the key given by aws to connect to the remote machine from our local machine. We used GitBash and Cygwin to connect to the remote machine over secured shell. After connecting we installed python, Jupyter, few of the python’s modules on the remote machine. We then setup a basic firewall to have a layer of security. 
After our remote machine was setup with the necessary installation, we then transferred the necessary files from local host to remote machine using ‘secured-copy (scp)’. The files included the below:

1.	Trained model as python object (model_fn.pth)
2.	The 10 image labels as pickle object (val_data_classes.pkl)
3.	The python script for making the predictions using request and flask (mlops-pj.py)
4.	Few sample images (cat.jpg, dog.jpg, deer.jpg, car.jpg)

After all the files were copied to remote machine, we used the ssh client to run the python script on the remote machine. This put our model into live production and was opened to the internet to access it using http requests. We will use the remote machine’s public IPv4 address to access it.

Few SSH commands that we used:<br>
ssh -i awsmachine-22.pem ubuntu@ip-address<br>
scp -i awsmachine-22.pem model_fn.pth ubuntu@ ip-address:<br>
vim mlopj-pj. py <br>
python mlopj-pj. py<br>

Below is a snapshot on how we accessed the model from our local machine and received predictions  from it.

![image](https://user-images.githubusercontent.com/35283246/163793952-35c873c5-2cde-45ba-a799-85982063d264.png)

## 2.	Deploying the Pytorch model with Flask and Heroku 

Link to our Heroku Application: https://ml-delpoy.herokuapp.com/

In this, we will create a flask app with a REST API that returns the result as json data and then we will deploy it to Heroku using our GitHub repository. The Heroku based application will allow us to upload an image from local machine through a web page and then return predictions on the uploaded image. Few of the technologies that we leveraged to pull this deployment are:

1.Flask
2. HTML
3. CSS 
4. Git
5. Heroku
6. Pytorch
7. GitBash

To deploy the model on Heroku, we needed additional setup up. First, in addition to all the necessary files, we need to create a ‘procfile’. Heroku apps include a ‘procfile’ that specifies the commands that are executed by the app on startup. We created a procfile with the command ‘web: gunicorn wsgi:app’ and saved it locally. Gunicorn stands for ‘Green Unicorn’. ‘Green Unicorn’ is a Python WSGI HTTP Server for UNIX and WSGI stands for Web Server Gateway Interface, and it is needed for Python web applications. This web server gateway helps us to serve static pages like HTML and CSS.

Second, we created our ‘requirements.txt’ which lists all the packages/modules that needs to be installed for the script to run. Some of the packages are torch, torchvision, gunicorn, numpy, pandas, flask, pickle.  

Third, we built two static webpages using HTML, that acts as the front interface for the Heroku app and allows users to upload an image for our model and get the predictions back.  We used CSS to style and layout the static webpage. We used ‘render_template’ function from flask to generate the html template containing the static data to be displayed.

All the required files for building the Heroku application are: 
Procfile, model_fn.pth, requirements.txt, val_data_classes.pkl, index.html, result.html, style.css

All the files were then uploaded to our GitHub repository. You can follow the link to see all the dependencies.

Challenge Faced :  While we were uploading the files to the repository, we faced a challenge in uploading the trained model parameters (model_fn.pth) to our GitHub account. GitHub allows upload up to a maximum size of 25mb over web. And since our ‘model_fn.pth’ was of 44mb, we were not able to upload it to our repository over web. To overcome this, we used GitBash CLI to first clone the  GitHub repository on our local machine. Then, we transferred the file ‘model_fn.pth’ to the locally created copy of the repository. After that we uploaded the entire repository from our local machine to GitHub. Few of the commands used are :
$ git add .
$ git commit -m "Add existing file"
$ git push origin your-branch

Next, we go to create Heroku and create a free-tier account with it. After creating the account we will create the application using ‘Create App’ functionality of Heroku. Heroku allows three different deployment methods for your application : 

•	Using Heroku CLI<br>
•	Using GitHub<br>
•	Using Container Registry<br>

We chose the second option. Using this, we linked our GitHub account with Heroku and chose the repository which had all the files we uploaded. Once linked, we have two options to choose for deploying our application, ‘Automatic Deploys’ and ‘Manual Deploys’. We chose manual deployment for the one time showcase. 
Challenge Faced :  Once you hit ‘deploy’ the Heroku runs the requirements.txt file from your linked repository and install all the mentioned modules. Now, two of the modules were ‘torch’ and ‘torchvision’. Since ‘torch’ is a big module of around 800mb and Heroku gives a container space of 500mb per deployment, we were unable to proceed with the deployment. Hence, to resolve this, we installed specific CPU only versions of torch and torchvision which is around 45mb:

torch==1.7.1+cpu
torchvision==0.8.2+cpu

After successful deployment, the application is now live on Heroku’s server, and we can access it using the link mentioned. 
Please note : The classifier has been trained on images from ten different classes. And hence, to allow it to make better predictions, please upload an image from the above mentioned ten classes. 

Below is a snapshot of landing page where we upload image:

![image](https://user-images.githubusercontent.com/35283246/163794233-e3c34f1a-3446-4f75-95ea-e5e36e3b2c8f.png)

Below is a snapshot of results page when we uploaded  image of a cat:

![image](https://user-images.githubusercontent.com/35283246/163794264-b8af2856-0605-4bdb-b905-078e84d2f2cf.png)

## 3.	Using GCP’s ‘Cloud Run’ to deploy the containerized docker image  (Zohaib)

Through this method we will demonstrate how to build a Docker image, containerize, and deploy it, and access it via HTTP requests using GCP ‘cloud run’ functionality. A few of the functionalities that we explored are:

•	Google Cloud Platform
•	GCP API
•	GCP Cloud Run
•	Google Cloud SDK and CLI
•	Docker images and containers

Below are the outlined steps:

A.	Write App (Flask and Pytorch) :<br>
Here we used the same components (python script, saved model, requirements file) that we used while building  web application using Heroku.

B.	Setting Up Cloud:<br>
In this step, we first created an account using the free tier on google cloud platform. Next, we created a new project and selecting all the necessary details like name, region, etc. After that, we had to activate two APIs of GCP i.e. cloud run api and cloud build api. Google Cloud APIs are programmatic interfaces to Google Cloud Platform services. These api allows us to use the power of gcp computing with different applications we build.

C.	Installing and Initializing Google Cloud SDK:<br>

Using this step, we installed the google cloud’s software development kit (SDK) on our local machine. This kit is a set of tools that are used to manage applications and resources that are hosted on google cloud platform.
After installing, we will use the gcp CLI on our local machine to initialize it. We use the command $ gcloud init. This command  sets the account property in the configuration to the specified account.

D.	Creating dockerfile, .dockerignore:<br>

Here, we will create three different files to be used as part of the docker image. First, the docker file is created in the same directory on local machine where all our files are located.
Few lines of command from our dockerfile are: 
 FROM python:3.9-slim
 ENV PYTHONUNBUFFERED True
 ENV APP_HOME /app   => ‘app’ indicates the flask app that we created
 WORKDIR $APP_HOME
 COPY . ./
 RUN pip install -r requirements.txt

Next, we will create ‘.dockerignore’. ‘.dockerignore’ file is used to ignore files and folders when you try to build a Docker Image. We specified the list of files and directories inside the ‘.dockerignore’ file.
 
E.	GCP Cloud Run build and Deploy:<br>

Now after all our files were build and saved at one location. We will build and deploy the docker container over GCP cloud run. For this purpose we will use two different commands. Below, first command containerizes the docker image and the second command deploy the container to cloud run.
$ gcloud builds submit --tag gcr.io/deploy-ml-342403/predict
$ gcloud run deploy --image gcr.io/ deploy-ml-342403/predict --platform managed

Challenge Faced: When you run the first command it asks for two sequential inputs from  the user. First is the region where my GCP account is hosted. We need to select this from the given options, and it needs to be correct. Hence, we had to explore our gcp account to find our server’s region. Second is the name of the container that is deployed. Please note that the function name doesn’t accept any ‘_’.
The ‘deploy-ml-342403’ is the name of my project on GCP and ‘predict’ is the flask gateway function in my script. After the two commands ran successfully, we got a HTTP link to connect to the containerized model that was deployed over cloud. Below is the screenshot of the deployed container: 









