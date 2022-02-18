# ML-Deployement

Here I built an ML based Python API hosted on Heroku. 

I made a cloud API with a Restnet18 image classifier that works on image classes of CIFAR-10. The Resnet18 model was trained locally and the trained parameters are then saved. The model is called and loaded each time a POST request is made using the Heroku App.
