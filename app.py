from torchvision import models, transforms
import torch.nn as nn
import torch
import streamlit as st
from PIL import Image
import torch


def load_image(image_file):
    img = Image.open(image_file)
    return img


def get_available_device():
    device_name = 'cpu'
    if torch.cuda.is_available():  # NVIDIA GPU
        device_name = 'cuda'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():  # Apple Silicon GPU
        device_name = 'mps'
    return torch.device(device_name)


st.subheader("Image")
image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

print(type(image_file))

if image_file is not None:

    st.image(load_image(image_file), width=250)
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(image_file)
    batch_t = torch.unsqueeze(transform(img), 0)

    # Creating an instance of the model and change the structure according to the trained one
    model = models.vgg16()
    model.classifier = nn.Sequential(
        nn.Linear(in_features=512, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=512, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=512, out_features=2, bias=True)
    )
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    # Loading the parameters from the trained model
    model.load_state_dict(torch.load('./checkpoints/conv_layers_frozen.pth'))
    model.eval()

    device = get_available_device() # get any available device for hardware accelaration

    with torch.no_grad():
        model = model.to(device=device)
        batch_t = batch_t.to(device=device)
        output = model(batch_t)
        pred = output.argmax(dim=1)
        if pred == 0:
            st.write('The image contains St. George')
        elif pred == 1:
            st.write('The image does not contain St. George')

# %%
