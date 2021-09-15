#Marcus Quirk

from PIL import Image
import numpy as np
import pathlib
import os

#Define constants for preprocessing
#Hardcoded filepaths can be changed accordingly
size = 200,200
filepath = "./bird_images/test/"
save_filepath = "./bird_images/test_resized/"
directory = pathlib.Path(filepath)

#Create a list of all the categories (i.e. species of bird)
categories = os.listdir(filepath)

#Iterate through each directory and create read and write filepaths
for category in categories:
    cur_filepath = filepath + category
    cur_save = save_filepath + category
    os.mkdir(cur_save)
    files = os.listdir(cur_filepath)
    
    #Iterate through the files in the read filepath
    #Resize and convert each image to greyscale before saving to the write directory
    for file in files:
        image = Image.open(cur_filepath + '/' + file).convert('L')
        image = image.resize(size)
        image.save(cur_save + '/' + file)
