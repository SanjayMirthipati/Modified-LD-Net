#Importing libraries

import os
import sys
import glob
import cv2
import math
import numpy
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
from torchvision import transforms
from torchsummary import summary
from PIL import Image
from matplotlib import pyplot as plt

# LD_Net

class LightDehaze_Net(nn.Module):

	def __init__(self):
		super(LightDehaze_Net, self).__init__()

		# LightDehazeNet Architecture
		self.relu = nn.ReLU(inplace=True)

		self.e_conv_layer1 = nn.Conv2d(3,8,1,1,0,bias=True)
		self.e_conv_layer2 = nn.Conv2d(8,8,3,1,1,bias=True)
		self.e_conv_layer3 = nn.Conv2d(8,8,5,1,2,bias=True)
		self.e_conv_layer4 = nn.Conv2d(16,16,7,1,3,bias=True)
		self.e_conv_layer5 = nn.Conv2d(16,16,3,1,1,bias=True)
		self.e_conv_layer6 = nn.Conv2d(16,16,3,1,1,bias=True)
		self.e_conv_layer7 = nn.Conv2d(32,32,3,1,1,bias=True)
		self.e_conv_layer8 = nn.Conv2d(32,32,3,1,1,bias=True)
		self.e_conv_layer9 = nn.Conv2d(88,3,3,1,1,bias=True)


	def forward(self, img):
		pipeline = []
		pipeline.append(img)

		conv_layer1 = self.relu(self.e_conv_layer1(img))
		conv_layer2 = self.relu(self.e_conv_layer2(conv_layer1))
		conv_layer3 = self.relu(self.e_conv_layer3(conv_layer2))

		# concatenating conv1 and conv3
		concat_layer1 = torch.cat((conv_layer1,conv_layer3), 1)

		conv_layer4 = self.relu(self.e_conv_layer4(concat_layer1))
		conv_layer5 = self.relu(self.e_conv_layer5(conv_layer4))
		conv_layer6 = self.relu(self.e_conv_layer6(conv_layer5))

		# concatenating conv4 and conv6
		concat_layer2 = torch.cat((conv_layer4, conv_layer6), 1)

		conv_layer7= self.relu(self.e_conv_layer7(concat_layer2))
		conv_layer8 = self.relu(self.e_conv_layer8(conv_layer7))

		# concatenating conv2, conv5, and conv8
		concat_layer3 = torch.cat((conv_layer2,conv_layer5,conv_layer7,conv_layer8),1)

		conv_layer9 = self.relu(self.e_conv_layer9(concat_layer3))


		dehaze_image = self.relu((conv_layer9 * img) - conv_layer9 + 1)
		#J(x) = clean_image, k(x) = x8, I(x) = x, b = 1
		# J(x) = (x8 * x) -x8 + b
		return dehaze_image


# Assuming you have a CUDA-enabled GPU, check if it is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the appropriate device (CPU or GPU)
model = LightDehaze_Net().to(device)

input_size = (3, 640, 480)

# Generate dummy input data on the same device as the model
dummy_input = torch.randn(1, *input_size).to(device)

# Call the summary function with the dummy input
summary(model, input_size)

def image_haze_removal(input_image):
    # Convert input image to hazy tensor
    hazy_image = transforms.ToTensor()(input_image).unsqueeze(0)

    # Load the trained model
    ld_net = LightDehaze_Net()
    ld_net.load_state_dict(torch.load('/content/drive/MyDrive/Modified LD_Net/trained_LD_Net (1).pth'))
    ld_net.eval()

    # Perform dehazing
    with torch.no_grad():
        dehaze_image = ld_net(hazy_image)

    # Convert the dehazed tensor to PIL Image
    dehaze_image = transforms.ToPILImage()(dehaze_image.squeeze(0))

    return dehaze_image

#For testing
def calculate_psnr(image1, image2):
    mse = np.mean((np.array(image1) - np.array(image2)) ** 2)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def single_dehaze_test(input, ground_truth):
    input_image = cv2.imread(input)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(input_image)

    # Perform dehazing
    start_time = time.time()
    dehaze_image = image_haze_removal(input_image)
    end_time = time.time()
    print("Time taken for dehazing:", end_time - start_time)

    dehaze_image.save('/content/drive/MyDrive/Modified LD_Net/1/dehazed_image.jpg')

    # Load the ground truth image
    gt_image = Image.open(ground_truth)

    # Calculate PSNR
    psnr = calculate_psnr(dehaze_image, gt_image)
    print("PSNR:", psnr)

    # Display the input and dehazed images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(input_image)
    plt.axis('off')
    plt.title('Input Image')

    plt.subplot(1, 3, 2)
    plt.imshow(dehaze_image)
    plt.axis('off')
    plt.title('Dehazed Image')

    plt.subplot(1, 3, 3)
    plt.imshow(gt_image)
    plt.axis('off')
    plt.title('Ground Truth Image')

    plt.show()

input = '/content/drive/MyDrive/archive/haze/0473_0.9_0.2.jpg'
ground_truth = '/content/drive/MyDrive/archive/clear_images/0473.jpg'
single_dehaze_test(input, ground_truth)