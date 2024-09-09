#Importing libraries

import os
import sys
import glob
import random
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

random.seed(113)

import torch
import torch.nn as nn

# Define the custom activation function
def shid(x):
    return x * torch.sin(torch.sigmoid(x))

# Create a custom activation module
class Sep(nn.Module):
    def __init__(self):
        super(Sep, self).__init__()

    def forward(self, x):
        return shid(x)

# Define the LightDehaze_Net model with the custom activation function
class LightDehaze_Net(nn.Module):

    def __init__(self):
        super(LightDehaze_Net, self).__init__()

        # LightDehazeNet Architecture
        self.sep = Sep()

        self.e_conv_layer1 = nn.Conv2d(3, 8, 1, 1, 0, bias=True)
        self.e_conv_layer2 = nn.Conv2d(8, 8, 3, 1, 1, bias=True)
        self.e_conv_layer3 = nn.Conv2d(8, 8, 5, 1, 2, bias=True)
        self.e_conv_layer4 = nn.Conv2d(16, 16, 7, 1, 3, bias=True)
        self.e_conv_layer5 = nn.Conv2d(16, 16, 3, 1, 1, bias=True)
        self.e_conv_layer6 = nn.Conv2d(16, 16, 3, 1, 1, bias=True)
        self.e_conv_layer7 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv_layer8 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv_layer9 = nn.Conv2d(88, 3, 3, 1, 1, bias=True)

    def forward(self, img):
        pipeline = []
        pipeline.append(img)

        conv_layer1 = self.sep(self.e_conv_layer1(img))
        conv_layer2 = self.sep(self.e_conv_layer2(conv_layer1))
        conv_layer3 = self.sep(self.e_conv_layer3(conv_layer2))

        # concatenating conv1 and conv3
        concat_layer1 = torch.cat((conv_layer1, conv_layer3), 1)

        conv_layer4 = self.sep(self.e_conv_layer4(concat_layer1))
        conv_layer5 = self.sep(self.e_conv_layer5(conv_layer4))
        conv_layer6 = self.sep(self.e_conv_layer6(conv_layer5))

        # concatenating conv4 and conv6
        concat_layer2 = torch.cat((conv_layer4, conv_layer6), 1)

        conv_layer7 = self.sep(self.e_conv_layer7(concat_layer2))
        conv_layer8 = self.sep(self.e_conv_layer8(conv_layer7))

        # concatenating conv2, conv5, conv7, and conv8
        concat_layer3 = torch.cat((conv_layer2, conv_layer5, conv_layer7, conv_layer8), 1)

        conv_layer9 = self.sep(self.e_conv_layer9(concat_layer3))

        dehaze_image = self.sep((conv_layer9 * img) - conv_layer9 + 1)
        # J(x) = clean_image, k(x) = x8, I(x) = x, b = 1
        # J(x) = (x8 * x) - x8 + b
        return dehaze_image

# Example usage
model = LightDehaze_Net()
print(model)

import torch
from torchsummary import summary

# Assuming you have a CUDA-enabled GPU, check if it is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the appropriate device (CPU or GPU)
model = LightDehaze_Net().to(device)

input_size = (3, 640, 480)

# Generate dummy input data on the same device as the model
dummy_input = torch.randn(1, *input_size).to(device)

# Call the summary function with the dummy input
summary(model, input_size)

# Preparing data

def preparing_training_data(hazefree_images_dir, hazeeffected_images_dir):


	train_data = []
	validation_data = []

	hazy_data = glob.glob(hazeeffected_images_dir + "*.jpg")

	data_holder = {}

	for h_image in hazy_data:
		h_image = h_image.split("/")[-1]
		id_ = h_image.split("_")[0] + ".jpg"
		if id_ in data_holder.keys():
			data_holder[id_].append(h_image)
		else:
			data_holder[id_] = []
			data_holder[id_].append(h_image)


	train_ids = []
	val_ids = []

	num_of_ids = len(data_holder.keys())
	for i in range(num_of_ids):
		if i < num_of_ids*9/10:
			train_ids.append(list(data_holder.keys())[i])
		else:
			val_ids.append(list(data_holder.keys())[i])


	for id_ in list(data_holder.keys()):

		if id_ in train_ids:
			for hazy_image in data_holder[id_]:

				train_data.append([hazefree_images_dir + id_, hazeeffected_images_dir + hazy_image])


		else:
			for hazy_image in data_holder[id_]:

				validation_data.append([hazefree_images_dir + id_, hazeeffected_images_dir + hazy_image])



	random.shuffle(train_data)
	random.shuffle(validation_data)

	return train_data, validation_data


class hazy_data_loader(data.Dataset):

	def __init__(self, hazefree_images_dir, hazeeffected_images_dir, mode='train'):

		self.train_data, self.validation_data = preparing_training_data('/content/drive/MyDrive/archive/clear_images/', '/content/drive/MyDrive/archive/haze/')

		if mode == 'train':
			self.data_dict = self.train_data
			print("Number of Training Images:", len(self.train_data))
		else:
			self.data_dict = self.validation_data
			print("Number of Validation Images:", len(self.validation_data))



	def __getitem__(self, index):

		hazefree_image_path, hazy_image_path = self.data_dict[index]

		hazefree_image = Image.open(hazefree_image_path)
		hazy_image = Image.open(hazy_image_path)

		hazefree_image = hazefree_image.resize((480,640), Image.LANCZOS)
		hazy_image = hazy_image.resize((480,640), Image.LANCZOS)

		hazefree_image = (np.asarray(hazefree_image)/255.0)
		hazy_image = (np.asarray(hazy_image)/255.0)

		hazefree_image = torch.from_numpy(hazefree_image).float()
		hazy_image = torch.from_numpy(hazy_image).float()

		return hazefree_image.permute(2,0,1), hazy_image.permute(2,0,1)

	def __len__(self):
		return len(self.data_dict)

from google.colab import drive
drive.mount('/content/drive')



!pip install pynvml


import time
import psutil
import torch
import torch.nn as nn
import csv
from datetime import datetime

try:
    import pynvml
    pynvml.nvmlInit()
    pynvml_available = True
except ImportError:
    pynvml_available = False

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def log_system_usage(log_file, epoch, batch_idx, loss, elapsed_time, learning_rate, gradient_norm, batch_processing_time):
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().used / (1024 ** 3)  # Convert to GB

    gpu_memory_allocated = gpu_memory_reserved = gpu_utilization = 0
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
        if pynvml_available:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_memory_allocated = mem_info.used / (1024 ** 3)  # Convert to GB
            gpu_memory_reserved = mem_info.total / (1024 ** 3)  # Convert to GB
            gpu_utilization = util_rates.gpu

    with open(log_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), epoch, batch_idx, loss, elapsed_time, learning_rate, gradient_norm,
                         batch_processing_time, cpu_usage, ram_usage, gpu_memory_allocated, gpu_memory_reserved, gpu_utilization])
    ti = datetime.now()
    # Print system usage
    print(f"Timestamp: {datetime.now()}")
    print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss:.4f}, Time: {elapsed_time:.2f}s")
    print(f"Learning Rate: {learning_rate}, Gradient Norm: {gradient_norm:.4f}, Batch Processing Time: {batch_processing_time:.2f}s")
    print(f"CPU Usage: {cpu_usage:.2f}%")
    print(f"RAM Usage: {ram_usage:.2f} GB")
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} GB")
        print(f"GPU Memory Reserved: {gpu_memory_reserved:.2f} GB")
        print(f"GPU Utilization: {gpu_utilization:.2f}%")
    print("-" * 40)

def train(args):
    ti = datetime.now()
    log_file = "/content/drive/MyDrive/Classroom/training_log_10k_sep"+str(ti)+".csv"

    # Create log file and write headers
    with open(log_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Epoch", "Batch", "Loss", "Elapsed Time (s)", "Learning Rate", "Gradient Norm",
                         "Batch Processing Time (s)", "CPU Usage (%)", "RAM Usage (GB)", "GPU Memory Allocated (GB)",
                         "GPU Memory Reserved (GB)", "GPU Utilization (%)"])

    #ld_net = LightDehaze_Net().cuda()
    #ld_net.apply(weights_init)
    ld_net = LightDehaze_Net().cuda()
    ld_net.load_state_dict(torch.load('/content/drive/MyDrive/Classroom/New_weights_10K_sep/Epoch0_sep_2024-06-02 16:56:26.202471.pth'))

    training_data = hazy_data_loader(args["train_original"], args["train_hazy"])
    validation_data = hazy_data_loader(args["train_original"], args["train_hazy"], mode="val")
    training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(ld_net.parameters(), lr=float(args["learning_rate"]), weight_decay=0.0001)

    ld_net.train()

    num_of_epochs = int(args["epochs"])
    total_batches = len(training_data_loader)
    total_training_time = 0
    for epoch in range(num_of_epochs):
        current_epoch = epoch + 1
        print(f"Epoch {current_epoch}/{num_of_epochs}")

        training_loss = 0.0
        epoch_start_time = time.time()
        for batch_idx, (hazefree_image, hazy_image) in enumerate(training_data_loader, 1):
            batch_start_time = time.time()

            hazefree_image = hazefree_image.cuda()
            hazy_image = hazy_image.cuda()

            dehaze_image = ld_net(hazy_image)

            loss = criterion(dehaze_image, hazefree_image)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ld_net.parameters(), 0.1)
            optimizer.step()

            training_loss += loss.item()

            batch_processing_time = time.time() - batch_start_time
            total_training_time += batch_processing_time
            learning_rate = optimizer.param_groups[0]['lr']
            gradient_norm = torch.nn.utils.clip_grad_norm_(ld_net.parameters(), 0.1).item()

            elapsed_time = time.time() - epoch_start_time

            print(f"\rBatch [{batch_idx}/{total_batches}], Loss: {loss.item():.4f}, Time: {elapsed_time:.2f}s, ", end="")

            # Log system usage
            log_system_usage(log_file, current_epoch, batch_idx, loss.item(), elapsed_time, learning_rate, gradient_norm, batch_processing_time)

        avg_training_loss = training_loss / total_batches
        print("\nTraining Loss:", avg_training_loss)

        # Rest of the code for validation and saving the model
        for iter_val, (hazefree_image, hazy_image) in enumerate(validation_data_loader):
            hazefree_image = hazefree_image.cuda()
            hazy_image = hazy_image.cuda()

            dehaze_image = ld_net(hazy_image)
            torch.save(ld_net.state_dict(), "/content/drive/MyDrive/Classroom/New_weights_10K/" + "Epoch" + str(epoch) + '_sep_'+str(ti)+'.pth')

        torch.save(ld_net.state_dict(), "/content/drive/MyDrive/Classroom/New_weights_10K/" + "trained_LD_Net_10k_sep_"+str(ti)+".pth")

args = {
    "train_hazy": "/content/drive/MyDrive/archive/haze/",
    "train_original": "/content/drive/MyDrive/archive/clear_images/",
    "epochs": "30",
    "learning_rate": "0.001"
}

train(args)

def image_haze_removal(input_image):
    # Convert input image to hazy tensor
    hazy_image = transforms.ToTensor()(input_image).unsqueeze(0)

    # Load the trained model
    ld_net = LightDehaze_Net()
    ld_net.load_state_dict(torch.load('/content/drive/MyDrive/Modified LD-Net/Modified LD-Net/Trained weights/Trained_LDNet.pth'))
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
    # Read input image using OpenCV
    input_image = cv2.imread(input)
    #input_image = transforms.ToPILImage()(input_image.squeeze(0))
    #input_image.save('/content/drive/MyDrive/Modified LD_Net/1/input_image.jpg')
    # Convert OpenCV image to PIL Image
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(input_image)

    # Perform dehazing
    dehaze_image = image_haze_removal(input_image)
    #dehaze_image.save('/content/drive/MyDrive/Modified LD_Net/1/dehafrzed_image.jpg')

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

input = '/content/drive/MyDrive/Modified LD_Net/0473_0.9_0.2.jpg'
ground_truth = '/content/drive/MyDrive/Modified LD_Net/0473.jpg'
single_dehaze_test(input, ground_truth)

import os
import random

# Function to select a random image from a directory
def select_random_image(directory):
    images = [file for file in os.listdir(directory) if file.endswith(('jpg', 'jpeg', 'png'))]
    random_image = random.choice(images)
    return os.path.join(directory, random_image)

# Updated single_dehaze_test function with random image selection
def single_dehaze_test(input_dir, ground_truth_dir, csv_path):
    # Select a random input image
    input_path = select_random_image(input_dir)

    # Corresponding ground truth image
    gt_image_name = os.path.splitext(os.path.basename(input_path))[0] + '.jpg'
    ground_truth_path = os.path.join(ground_truth_dir, gt_image_name)

    # Read input image using OpenCV
    input_image = cv2.imread(input_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(input_image)

    # Perform dehazing
    dehaze_image = image_haze_removal(input_image)

    # Load the ground truth image
    gt_image = Image.open(ground_truth_path)

    # Calculate PSNR
    psnr_value = calculate_psnr(dehaze_image, gt_image)

    # Calculate SSIM
    ssim_value = calculate_ssim(dehaze_image, gt_image)

    # Encode images to base64
    input_image_base64 = image_to_base64(input_image)
    dehaze_image_base64 = image_to_base64(dehaze_image)

    # Write PSNR, SSIM, and images to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['PSNR', 'SSIM', 'Input Image', 'Dehazed Image'])
        csvwriter.writerow([psnr_value, ssim_value, input_image_base64, dehaze_image_base64])

    print(f"Input Image: {input_path}")
    print(f"Ground Truth Image: {ground_truth_path}")
    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_value}")

    # Display the images
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

# Example usage
input_dir = '/content/drive/MyDrive/Modified LD_Net/input_images'
ground_truth_dir = '/content/drive/MyDrive/Modified LD_Net/ground_truth_images'
csv_path = '/content/drive/MyDrive/Modified LD_Net/results.csv'
single_dehaze_test(input_dir, ground_truth_dir, csv_path)