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


