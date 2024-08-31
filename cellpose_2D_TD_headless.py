#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q cellpose')
get_ipython().system('pip install -q ipywidgets')
get_ipython().system('pip install -q matplotlib --upgrade')


# In[ ]:


from cellpose import utils, io, models, plot
from ipywidgets import interact, interact_manual
import ipywidgets as widgets
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imread, imwrite
import time
import sys
import os, random
import torch
if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is NOT available! Are you on the right node?")
    exit()


# In[ ]:


import logging

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
log = logging.getLogger()
fhandler = logging.FileHandler(filename='cellpose_2D_log.txt', mode='w')
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
fhandler.setFormatter(formatter)
log.addHandler(fhandler)


# In[ ]:


#@markdown ### Provide the path to your dataset and to the folder where the predictions are saved, then play the cell to predict outputs from your unseen images.

Data_folder = "/research/rgs01/home/clusterHome/tdas/cellpose_input" #@param {type:"string"}
Result_folder = "/research/rgs01/home/clusterHome/tdas/cellpose_output" #@param {type:"string"}

#@markdown ###Are your data single images or stacks?

Data_type = "Single_Images" #@param ["Single_Images", "Stacks (2D + t)"]

#@markdown ###What model do you want to use?

model_choice = "Cytoplasm2" #@param ["Cytoplasm","Cytoplasm2", "Cytoplasm2_Omnipose", "Bacteria_Omnipose", "Nuclei", "Own_model"]

#@markdown ####If using your own model, please provide the path to the model (not the folder):

Prediction_model = "" #@param {type:"string"}

#@markdown ### What channel do you want to segment?

Channel_to_segment= "Green" #@param ["Grayscale", "Blue", "Green", "Red"]

# @markdown ###If you chose the model "cytoplasm" indicate if you also have a nuclear channel that can be used to aid the segmentation.

Nuclear_channel= "Blue" #@param ["None", "Blue", "Green", "Red"]

#@markdown ### Segmentation parameters:
Object_diameter =  250 #@param {type:"number"}

Flow_threshold = 0.4 #@param {type:"slider", min:0.1, max:1.1, step:0.1}

#cellprob_threshold = 0 #@param {type:"slider", min:-6, max:6, step:1}


# Find the number of channel in the input image

random_choice = random.choice(os.listdir(Data_folder))
x = io.imread(Data_folder+"/"+random_choice)
n_channel = 1 if x.ndim == 2 else x.shape[-1]

if Channel_to_segment == "Grayscale":
  segment_channel = 0

  if Data_type == "Single_Images":
    if not n_channel == 1:
        print(bcolors.WARNING +"!! WARNING: your image has more than one channel, choose which channel you want to use for your predictions !!")

if Channel_to_segment == "Blue":
  segment_channel = 3

if Channel_to_segment == "Green":
  segment_channel = 2

if Channel_to_segment == "Red":
  segment_channel = 1

if Nuclear_channel == "Blue":
  nuclear_channel = 3

if Nuclear_channel == "Green":
  nuclear_channel = 2

if Nuclear_channel == "Red":
  nuclear_channel = 1

if Nuclear_channel == "None":
  nuclear_channel = 0

if model_choice == "Cytoplasm":  
  channels=[segment_channel,nuclear_channel]
  model = models.Cellpose(gpu=True, model_type="cyto")
  print("Cytoplasm model enabled")

if model_choice == "Cytoplasm2":  
  channels=[segment_channel,nuclear_channel]
  model = models.Cellpose(gpu=True, model_type="cyto2")
  print("Cytoplasm2 model enabled")

if model_choice == "Cytoplasm2_Omnipose":  
  channels=[segment_channel,nuclear_channel]
  model = models.Cellpose(gpu=True, model_type="cyto2_omni")
  print("Cytoplasm2_Omnipose model enabled")
  
if model_choice == "Nuclei":
  channels=[segment_channel,0]
  model = models.Cellpose(gpu=True, model_type="nuclei")
  print("Nuclei model enabled")

if model_choice == "Bacteria_Omnipose":
  channels=[segment_channel,nuclear_channel]
  model = models.Cellpose(gpu=True, model_type="bact_omni")
  Object_diameter =  0
  print("Bacteria_omnipose model enabled")

if model_choice == "Own_model":
  channels=[segment_channel,nuclear_channel]
  model = models.CellposeModel(gpu=True, pretrained_model=Prediction_model, torch=True, diam_mean=30.0, net_avg=True, device=None, residual_on=True, style_on=True, concatenation=False)

  print("Own model enabled")

if Object_diameter == 0:
  Object_diameter = None
  print("The cell size will be estimated automatically for each image")

if Data_type == "Single_Images" :

  print('--------------------------------------------------------------')
#   @interact
#   def preview_results(file = os.listdir(Data_folder)):
#     source_image = io.imread(os.path.join(Data_folder, file))
    
#     if model_choice == "Own_model":
#       masks, flows, styles = model.eval(source_image, diameter=Object_diameter, flow_threshold=Flow_threshold, channels=channels)
    
#     else:
#       masks, flows, styles, diams = model.eval(source_image, diameter=Object_diameter, flow_threshold=Flow_threshold, channels=channels)
    
#     flowi = flows[0]
#     fig = plt.figure(figsize=(20,10))
#     plot.show_segmentation(fig, source_image, masks, flowi, channels=channels)
#     plt.tight_layout()
#     plt.show()


#   def batch_process():
  
  print("Your images are now beeing processed")

  for name in os.listdir(Data_folder):
    print("Performing prediction on: "+name)
    image = io.imread(Data_folder+"/"+name)
    short_name = os.path.splitext(name)

    if model_choice == "Own_model":
      masks, flows, styles = model.eval(image, diameter=Object_diameter, flow_threshold=Flow_threshold, channels=channels)
    else:
      masks, flows, styles, diams = model.eval(image, diameter=Object_diameter, flow_threshold=Flow_threshold, channels=channels)

    os.chdir(Result_folder)
    imwrite(str(short_name[0])+"_mask.tif", masks)

#   im = interact_manual(batch_process)
#   im.widget.children[0].description = 'Process your images'
#   im.widget.children[0].style.button_color = 'yellow'
#   display(im)

# if Data_type == "Stacks (2D + t)" :
#   print("Stacks (2D + t) are now beeing predicted")
  
#   print('--------------------------------------------------------------')
#   @interact
#   def preview_results_stacks(file = os.listdir(Data_folder)):
#     timelapse = imread(Data_folder+"/"+file)

#     if model_choice == "Own_model":
#       masks, flows, styles = model.eval(timelapse[0], diameter=Object_diameter, flow_threshold=Flow_threshold, channels=channels)
#     else:
#       masks, flows, styles, diams = model.eval(timelapse[0], diameter=Object_diameter, flow_threshold=Flow_threshold, channels=channels)
           
#     flowi = flows[0]
#     fig = plt.figure(figsize=(20,10))
#     plot.show_segmentation(fig, timelapse[0], masks, flowi, channels=channels)
#     plt.tight_layout()
#     plt.show()

#   def batch_process_stack():
#       print("Your images are now beeing processed")  
#       for image in os.listdir(Data_folder):
#         print("Performing prediction on: "+image)
#         timelapse = imread(Data_folder+"/"+image)
#         short_name = os.path.splitext(image)
#         n_timepoint = timelapse.shape[0]
#         prediction_stack = np.zeros((n_timepoint, timelapse.shape[1], timelapse.shape[2]))
        
#         for t in range(n_timepoint):
#           print("Frame number: "+str(t))
#           img_t = timelapse[t]

#           if model_choice == "Own_model":
#             masks, flows, styles = model.eval(img_t, diameter=Object_diameter, flow_threshold=Flow_threshold, channels=channels)
#           else:
#             masks, flows, styles, diams = model.eval(img_t, diameter=Object_diameter, flow_threshold=Flow_threshold, channels=channels)
            
              
#           prediction_stack[t] = masks
      
#         prediction_stack_32 = img_as_float32(prediction_stack, force_copy=False)
#         os.chdir(Result_folder)
#         imwrite(str(short_name[0])+".tif", prediction_stack_32)
  
#   im = interact_manual(batch_process_stack)
#   im.widget.children[0].description = 'Process your images'
#   im.widget.children[0].style.button_color = 'yellow'
#   display(im)          
     


# In[ ]:




