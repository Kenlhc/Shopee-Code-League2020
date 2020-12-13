# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# defining global variable path
image_path = "C:/Users/user/Desktop/Shopee_Code_League_Stuff/shopee-product-detection-dataset/train/train/"

for index in range(10, 42):
    final_image_path = image_path + str(index)
    for filename in os.listdir(final_image_path):
        if filename.endswith(".jpg"):
            ima = Image.open(os.path.join(final_image_path,filename)).convert('L')
            ima = ima.resize((100,100))
            new_path = "C:/Users/user/Desktop/Shopee_Code_League_Stuff/shopee-product-detection-dataset/train/train/" + str(index) + "_processed"
            ima.save(os.path.join(new_path,filename))    