

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
from PIL import Image
import pickle


# **Defining the transformations to perform on Images**

# In[ ]:


batch_size = 128

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# #### A) Download the CIFAR 10 dataset

# **Load and Transform the dataset using TorchVision**

# In[ ]:


train_data = datasets.CIFAR10('./data',
                              train = True,
                             download = True,
                              transform = train_transform
                             )
valid_data = datasets.CIFAR10('./data',
                              train = False,
                             download = True,
                              transform = valid_transform
                             )


# In[ ]:


train_data_classes = train_data.classes
print(train_data_classes)
val_data_classes = valid_data.classes
print(val_data_classes)


# In[ ]:


with open(os.path.join(sys.path[0], "val_data_classes.pkl"), 'wb') as f:
    pickle.dump(val_data_classes, f) 




# #### B) Use the pretrained Resnet18 model (from trochvision) to extract features. Use the features as inputs in a new multi-class logistic regression model (use nn.Linear/ nn.Module to define your model) -(a) Describe any choices made and report test performance. -(b) Display the top 5 correct predictions and the top 5 incorrect predictions in each class (show the images and the prediction labels) compactly.

# **Loading the Data for Training**

# In[ ]:


trainloader = torch.utils.data.DataLoader(train_data, batch_size,
                                        shuffle=True, num_workers=2,pin_memory=True)


testloader = torch.utils.data.DataLoader(valid_data, batch_size,
                                        shuffle=False, num_workers=2,pin_memory=True)


# In[ ]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device


# **Defining general training Model**

# In[ ]:


def train_model(model, loss_function, optimizer, data_loader):
    model.train()
    
    current_loss = 0
    current_acc = 0
    
    for i, (inputs, labels) in enumerate(data_loader):
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            
            loss = loss_function(outputs, labels)
            
            loss.backward()
            
            optimizer.step()
            
        current_loss += loss.item() * inputs.size(0)
        
        current_acc += torch.sum(predicted == labels.data)
        
    total_loss = current_loss / len(data_loader.dataset)
    total_acc = 100 * (current_acc.double() / len(data_loader.dataset))
    
    print('Train Loss: {:.4f}, Accuracy: {:.4f}%'.format(total_loss, total_acc))


# **Defining general Validation Model**

# In[ ]:


def val_model(model, loss_function, data_loader):
    
    model.eval()
    
    current_loss = 0
    current_acc = 0
    best_acc = 0.0
    
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            
            loss = loss_function(outputs, labels)
            
        current_loss += loss.item() * inputs.size(0)
        
        current_acc += torch.sum(predicted == labels.data)
        
    total_loss = current_loss / len(data_loader.dataset)
    total_acc = 100 * (current_acc.double() / len(data_loader.dataset))
    if total_acc > best_acc:
        best_acc = total_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    
    print('Validation Loss: {:.4f}, Accuracy: {:.4f}%'.format(total_loss, total_acc))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# **CNN as Feature Extractor**

# In[ ]:


epoch = 10

model = torchvision.models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 10)

model = model.to(device)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

for epch in range(epoch):
    
    print('Epoch {}/{}'.format(epch+1, epoch))
    train_model(model, loss_function, optimizer, trainloader)
    model_fe = val_model(model, loss_function, testloader)


# **Saving and Loading the Model**

# In[ ]:


torch.save(model_fe.state_dict(), os.path.join(sys.path[0], "model_fn.pth")) 


# **Store the Predictions using Feature Extractor** 

# In[ ]:


# def store_result(model):
#     was_training = model.training
#     model.eval()
#     column_names = ["inp_img","value", "pred_label", "act_label"]
#     df = pd.DataFrame(columns = column_names)
#     t=0
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(testloader):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             x, preds = torch.max(outputs, 1)
#             m = nn.Softmax(dim=1)
#             smo = m(outputs)    
#             for j in range(inputs.size()[0]):
#                 # images_so_far += 1
#                 df.loc[j+t,'inp_img'] = inputs.cpu().data[j].numpy()
#                 df.loc[j+t,'act_label'] = labels.cpu().data[j].numpy()
#                 df.loc[j+t,'pred_label'] = preds.cpu().data[j].numpy()
#                 df.loc[j+t,'value'] = smo.cpu().data[j].numpy().max()
#             t+=inputs.size()[0]
#         return df


# # In[ ]:


# df_fe=store_result(model_fe)


# # In[ ]:


# df_fe.head()


# In[ ]:


# airplane_df = df[df['act_label']==0].reset_index().sort_values(['value'],ascending=False)
# automobile_df = df[df['act_label']==1].reset_index().sort_values(['value'],ascending=False)
# bird_df = df[df['act_label']==2].reset_index().sort_values(['value'],ascending=False)
# cat_df = df[df['act_label']==3].reset_index().sort_values(['value'],ascending=False)
# deer_df = df[df['act_label']==4].reset_index().sort_values(['value'],ascending=False)
# dog_df = df[df['act_label']==5].reset_index().sort_values(['value'],ascending=False)
# frog_df = df[df['act_label']==6].reset_index().sort_values(['value'],ascending=False)
# horse_df = df[df['act_label']==7].reset_index().sort_values(['value'],ascending=False)
# ship_df = df[df['act_label']==8].reset_index().sort_values(['value'],ascending=False)
# truck_df = df[df['act_label']==9].reset_index().sort_values(['value'],ascending=False)


# #### General Visualization function for displaying Top 5 and Bottom 5 predictions of each class

# In[ ]:


# def visualize_top_bottom_results(df):
#     df=df.sort_values(['value'],ascending=False)
# #     print('Actual Class: ',val_data_classes[df.iloc[0,3]])
#     fig = plt.figure(figsize=(20,20))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     for j in range(5):
#         x = df.iloc[j,1].transpose(1,2,0)
#         x = std * x + mean
#         x = np.clip(x, 0, 1)
#         ax = plt.subplot(10, 2, j+1)
#         ax.axis('off')
#         lab = val_data_classes[df.iloc[j,3]]
#         ax.set_title('Correct prediction, predicted: {}'.format(lab))
#         plt.imshow(x)
#     df = df[df['act_label']!=df['pred_label']]
#     df=df.sort_values(['value'],ascending=True)
#     for j in range(5):
#         x = df.iloc[j,1].transpose(1,2,0)
#         x = std * x + mean
#         x = np.clip(x, 0, 1)
#         ax = plt.subplot(10, 2, j+1+5)
#         ax.axis('off')
#         lab = val_data_classes[df.iloc[j,3]]
#         ax.set_title('Incorrect prediction, Predicted : {}'.format(lab))
#         plt.imshow(x)
# #         print("\n")
# #         print('Actual Class: ',val_data_classes[df.iloc[0,3]])


# # #### Visualizing top and bottom predictions using Feature Extractor CNN

# # In[ ]:


# for i in range(10):
# #     print('Actual Class: ',val_data_classes[i])
#     visualize_top_bottom_results(df_fe[df_fe['act_label']==i].reset_index())





