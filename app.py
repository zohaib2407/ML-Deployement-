import io
from torchvision import datasets, models, transforms
from PIL import Image
import pickle
import torchvision
import torch.nn as nn
import torch
from flask import Flask, jsonify, request, render_template, request, redirect
import json
import os



def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    with open('val_data_classes.pkl', 'rb') as f:
        val_data_classes = pickle.load(f)
    model_fn = torchvision.models.resnet18()
    num_features = model_fn.fc.in_features
    model_fn.fc = nn.Linear(num_features, 10)
    model_fn.load_state_dict(torch.load('model_fn.pth'))
    model_fn.eval()
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model_fn.forward(tensor)
    _, y_hat = outputs.max(1)
    label = val_data_classes[y_hat]
    return y_hat.item(),label

app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def upload_file():
    # if request.method == 'POST':
    #     try:
    #         file = request.files['file']
    #         img_bytes = file.read()
    #         i,label = get_prediction(image_bytes=img_bytes)
    #         return render_template('result.html', class_id=i,class_name=label)
    #     except:
    #         return jsonify({'success': 'false', 'message': 'Input image was not passed correctly.'})
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return render_template('result.html', class_id=class_id,
                               class_name=class_name)
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))

