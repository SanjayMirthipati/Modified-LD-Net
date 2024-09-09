import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

# LightDehazeNet implementation
class LightDehaze_Net(nn.Module):
  def __init__(self):
    super(LightDehaze_Net, self).__init__()
        # Define the architecture
        # LightDehazeNet Architecture
    self.relu = nn.ReLU(inplace=True)
    self.e_conv_layer1 = nn.Conv2d(3,8,1,1,0,bias=True)
    self.e_conv_layer2 = nn.Conv2d(8,8,3,1,1,bias=True)
    self.e_conv_layer3 = nn.Conv2d(8,8,5,1,2,bias=True)
    self.e_conv_layer4 = nn.Conv2d(16,16,7,1,3,bias=True)
    self.e_conv_layer5 = nn.Conv2d(16,16,3,1,1,bias=True)
    self.e_conv_layer6 = nn.Conv2d(16,16,3,1,1,bias=True)
    self.e_conv_layer7 = nn.Conv2d(32,32,3,1,1,bias=True)
    self.e_conv_layer8 = nn.Conv2d(56,3,3,1,1,bias=True)
    self.e_conv_layer9 = nn.Conv2d(88,3,3,1,1,bias=True)

  def forward(self, img):
     # Define the forward pass
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


    # Save conv_layer9 output as an image
    output_image = transforms.ToPILImage()(conv_layer9.squeeze(0))
    #output_image.save('traffic/conlayer_output_image.png')

    dehaze_image = self.relu((conv_layer9 * img) - conv_layer9 + 1)
	  
    return dehaze_image


def image_haze_removal(input_image):
    # Convert input image to hazy tensor
    hazy_image = transforms.ToTensor()(input_image).unsqueeze(0)

    # Load the trained model
    ld_net = LightDehaze_Net()
    ld_net.load_state_dict(torch.load('weights/trained_LDNet.pth'))
    ld_net.eval()

    # Perform dehazing
    with torch.no_grad():
        dehaze_image = ld_net(hazy_image)

    # Convert the dehazed tensor to PIL Image
    dehaze_image = transforms.ToPILImage()(dehaze_image.squeeze(0))

    return dehaze_image


def calculate_psnr(image1, image2):
    mse = np.mean((np.array(image1) - np.array(image2)) ** 2)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    #psnr = 20 * np.log10(max_pixel) - 10 * np.sqrt(mse)
    return psnr

def single_dehaze_test(input, ground_truth):
    # Read input image using OpenCV
    input_image = cv2.imread(input)

    # Convert OpenCV image to PIL Image
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(input_image)

    # Perform dehazing
    dehaze_image = image_haze_removal(input_image)

    #saving the dehazed image
    dehaze_image.save("traffic/dehazed__new.jpg")

    # Calculate PSNR
    psnr = calculate_psnr(dehaze_image, gt_image)
    print("PSNR:", psnr)

input = 'archive/haze/0473_0.8_0.16.jpg'
ground_truth = 'archive/clear_images/0473.jpg'
single_dehaze_test(input, ground_truth)
