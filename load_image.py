from PIL import Image
import numpy as np
import pathlib
import os
##import csv

size = 200,200
filepath = "./bird_images/train/"
directory = pathlib.Path(filepath)

image_list = []

categories = list(os.walk(filepath))[0][1]

for category in categories:
    cur_filepath = filepath + category
    files = list(os.walk(cur_filepath))[0][2]
    for file in files:
        image = Image.open(cur_filepath + '/' + file).convert('L')
        image.resize(size)
        image_sequence = image.getdata()
        image_array = np.array(image_sequence)
        image_array = np.append(str(category), image_array)
        image_list.append(list(image_array))
    print('finished', str(category))
with open('birds_01.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(image_list)
