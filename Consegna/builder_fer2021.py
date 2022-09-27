import numpy as np
import pandas as pd
import math
import torch
from google.colab import drive


#mounting drive and reading the dataframe
drive.mount('/content/drive')

data = pd.read_csv('/content/drive/MyDrive/deep_learning/progetto/fer2013.csv')

#setting the size of image and how many filter we apply (im_d = #filter - 1)
im_d = 3
im_l = 48
im_h = 48

#function that apply the filter as lamba instruction
# image : the original image 
# filter : lambda instruction of the filter, it's building with (i,x,y), i as value of pixel, x and y the point on the filter matrix, x,y: 0<=x<l and 0<=y<l
# l : the size of filter
# OUT : new image as matrix, with dimensionality as original image 

def filterApply(image, filtro, l):
  newImage = torch.zeros(image.shape, dtype=image.dtype)
  z = w = 0
  for x in range(-1,image.shape[0]-l+1,1):
    for y in range(-1,image.shape[1]-l+1,1):
      for x1 in range(0,l):
        for y1 in range(0,l):
          newImage[z][w] += filtro(image[x+x1][y+y1] if (x+x1>=0 and y+y1>=0 and x<=image.shape[0]-l and y<=image.shape[1]-l) else 0,x1,y1)
      newImage[z][w] /= l*l
      w+=1
    z+=1
    w=0
  return newImage

#convert from string of pixel inside data["pixels"] to np-array and then in to "normalized" tensor
# x : data images array of strings
# filters : array of lambda filters
# kernel : array of kernel's size
#OUT : array that contains |x|*(|filters|+1) images, the original images and the filter, its shape is (|x|,|filter|+1, im_h, im_l), |filter|+1 is like im_d  

def arrToNpWFilter(x,filters,kernel):
    temp = []
    image = torch.zeros([0])
    for i,im in enumerate(x,0):
      image = torch.Tensor(np.array(im.split()).reshape(im_l,im_h, 1).astype('double')/255)
      tmp = [image]
      for i,f in enumerate(filters,0):
        tmp.append(filterApply(image,f,kernel[i]))      
      temp.append(torch.cat(tmp,0).view(im_d,im_l,im_h))
    return temp


#some traditional filter
sobol_filter = (lambda i,x,y: ((1-(x%2))*(1+(y%2))*(1-(x%3))*i +(1-(y%2))*(1+(x%2))*(1-(y%3))*i)/2)
vertical_filter = (lambda i,x,y: ((1-(y%2))*(1+(x%2))*(1-(y%3))*i))
horrizontal_filter = (lambda i,x,y: ((1-(x%2))*(1+(y%2))*(1-(x%3))*i))
contrast = (lambda i,x,y: (i*i))
high_contrast = (lambda i,x,y: (i*i*i))
low_contrast = lambda i,x,y: math.sqrt(i)

#convert, apply filter and rewrite all data
 
stro = ""

for idx,images in enumerate(arrToNpWFilter(data["pixels"],[sobol_filter,contrast],[3,2]), 0):
  stro = ','.join(str(x) for x in [int(i*255) for i in images.view(im_d*im_l*im_h)])
  data["pixels"][idx] = stro.replace(","," ")

#write a new dataframe
data.to_csv("fer2021.csv", index=False)