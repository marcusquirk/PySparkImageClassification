from PIL import Image
import numpy as np
import pathlib
import os


size = 200,200
filepath = "./bird_images/train/"
save_filepath = "./bird_images/train_resized/"
directory = pathlib.Path(filepath)

image_list = []

categories = list(os.walk(filepath))[0][1]

for category in categories:
    cur_filepath = filepath + category
    cur_save = save_filepath + category
    os.mkdir(cur_save)
    files = list(os.walk(cur_filepath))[0][2]
    for file in files:
        image = Image.open(cur_filepath + '/' + file).convert('L')
        image = image.resize(size)
        image.save(cur_save + '/' + file)
    print ('finished',str(category))
