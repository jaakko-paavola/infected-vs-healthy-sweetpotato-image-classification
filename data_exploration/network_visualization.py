""""
From https://github.com/utkuozbulak/pytorch-cnn-visualizations
"""

import sys
sys.path.append('/home/jarsba/git_opiskelu/data_science_project_1/Infected-sweetpotato-classification/models/model_factory.py')
sys.path.append('/home/jarsba/git_opiskelu/data_science_project_1/Infected-sweetpotato-classification/models/')
sys.path.append('/home/jarsba/git_opiskelu/data_science_project_1/Infected-sweetpotato-classification')

from dotenv import load_dotenv

import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import os
import copy
from PIL import Image
import matplotlib.cm as mpl_color_map
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

from models.model_factory import get_trained_model_by_id
from models.model_parts import ResNetBlock

load_dotenv()

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA')).resize((256, 256))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('../results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)




class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class LayerCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        
        if self.extractor.gradients is None:
            return None
        
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = guided_gradients
        weights[weights < 0] = 0 # discard negative gradients
        # Element-wise multiply the weight with its conv output and then, sum
        cam = np.sum(weights * target, axis=0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255

        return cam


if __name__ == '__main__':
    # Get params
    
    pretrained_model = get_trained_model_by_id("9zr4vtfh")
    modules = [module for module in pretrained_model.features.modules() if not isinstance(module, nn.Sequential)]
        
    for target_layer in range(0, 50):
        print(f"Target layer: {target_layer}")
        print(f"Target layer: {modules[target_layer]}")
        
        if isinstance(modules[target_layer], ResNetBlock):
            print(f"Skipping ResNet block")
            continue 
        
        original_image = Image.open(os.path.join(DATA_FOLDER_PATH, "Separated_plants/Trial_02/Dataset_01/Background_included/82-7-PS_Tray_419/plant_index_2.png")).convert('RGB')
        
        original_image_np = np.asarray(original_image)
        
        image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(50),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.09872966, 0.11726899, 0.06568969],
                                std=[0.1219357, 0.14506954, 0.08257045]),
        ])
        prep_img = image_transform(original_image_np)
        prep_img = prep_img.unsqueeze(0)
        prep_img = Variable(prep_img, requires_grad=True)

        target_class = 2
        file_name_to_export = f"restnet_layer{target_layer}"
        
        # Layer cam
        layer_cam = LayerCam(pretrained_model, target_layer=target_layer)
        # Generate cam mask
        cam = layer_cam.generate_cam(prep_img, target_class)
        
        if cam is None:
            print(f"Skipping layer {target_layer}")
            continue
        
        # Save mask
        save_class_activation_images(original_image, cam, file_name_to_export)
        print('Layer cam completed')