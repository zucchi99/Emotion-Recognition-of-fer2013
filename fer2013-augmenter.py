import numpy as np
import pandas as pd
import math
from enum import Enum
from tqdm.auto import tqdm
import imutils

class Filters(Enum) :
  ORIGINAL             = 0
  SOBEL                = 1
  VERTICAL             = 2
  HORIZONTAL           = 3
  CONTRAST_LOW         = 4
  CONTRAST_HIGH        = 5
  CONTRAST_VERY_HIGH   = 6
  FLIP_HORIZONTAL      = 7
  ROT_LEFT_60_DEGREES  = 8
  ROT_LEFT_40_DEGREES  = 9
  ROT_LEFT_20_DEGREES  = 10
  ROT_RIGHT_20_DEGREES = 11
  ROT_RIGHT_40_DEGREES = 12
  ROT_RIGHT_60_DEGREES = 13

class Fer2013_Augmenter :

  # pandas data df
  data = pd.DataFrame()
  
  # set of filters to be applied
  filters_to_use = {}
  # list of filters applied, same as before but in a different order
  # necessary to map each filter to the corresponding column
  filters_used_ordered = []
  
  # tuple of all possible emotions in dataset
  emotions = ('rage', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral')

  #setting the size of image and how many filter we apply (im_d = len(filters_to_use))
  im_d = 0
  im_l = 0
  im_h = 0

  # class intializer
  def __init__(self, data_df, filters, im_h=48, im_l=48) :
    # check params type consistency
    if not isinstance(filters, list) :
      raise Exception("filters must be a list of Filters(Enum)")
    if not isinstance(data_df, pd.core.frame.DataFrame) :
      raise Exception("data_df must be a pd.DataFrame()")
    for elem in filters :
      if not isinstance(elem, Filters) :
        raise Exception("filter unkown")
    # assign local variables
    self.im_h = im_h        
    self.im_l = im_l
    self.im_d = len(filters)
    self.data = data_df
    self.filters_to_use = set(filters)
        
    
  # dictionary of filters defined by lambdas transformation 
  # param (x,y): filter coordinates (NOT image)
  # param pix:   color value inside the pixel
  # returns:     apply a transformation to each pixel independently
  filters_lambda = {
    #'original'     : (lambda x, y, pix: (pix)),
    Filters.CONTRAST_LOW       : (lambda x, y, pix : (pix ** (1/2))),
    Filters.CONTRAST_HIGH      : (lambda x, y, pix : (pix ** 2)),
    Filters.CONTRAST_VERY_HIGH : (lambda x, y, pix : (pix ** 3)) 
  }


  # dictionary of filters defined by filter matrix
  filters_matrix = {
    Filters.HORIZONTAL : np.array([ [ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1] ]),
    Filters.VERTICAL   : np.array([ [ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1] ])
  }


  # param images: list of flat images as string delimited by space
  # returns:      list of images as np.array(im_h, im_l) of double
  def string_to_array(self, images):
    out = []
    for im in tqdm(images):
      # split()   --> converts text (delimited by space) to list of strings
      # reshape() --> reshapes a np.array to dimension given (from flat to 2D)
      # astype()  --> casts np.array of np.array of double
      out.append(self.string_to_array_image(im))
    return out


  # param image: list of flat images as string delimited by space
  # returns:     image as np.array(im_h, im_l) of double
  def string_to_array_image(self, image) :
    shape = (self.im_h, self.im_l)
    return np.array(image.split()).reshape(shape).astype('double')
      

  # param array: list/array to join
  # param sep:   list/array separator in the string (default space)
  # returns:     convert a list/array to a string delimited by separator 
  def array_to_string(self, array, sep=' ') :
    return sep.join(str(number) for number in array.astype(int))  

  
  # param image: image as np.array
  # returns:     image as np.array from 2D to 1D
  def flat_image(self, image) :
    return image.reshape(self.im_l * self.im_h)


  # param image: image as np.array
  # returns:     image as np.array from 1D to 2D
  def unflat_image(self, image) :
    return np.array([ np.array(image[(self.im_h * i):(self.im_h * (i+1))]) for i in range(self.im_l) ])
   

  # param image: image as np.array
  # returns:     image as np.array, a min max scaler is applied to image pixels, so they will be in range(0, 255)
  def scale_image_to_int_in_bounds(self, image) :
      # flat image to one-dimensional np.array
      flatted_image = self.flat_image(image)
      # get image bounds
      tot_max = max(flatted_image)
      tot_min = min(flatted_image)
      scaled = np.zeros(self.im_l * self.im_h)
      # check if image has a constant color
      if (tot_min == tot_max) :
          #image of just one constant color
          constant = tot_min
          if (constant > 0 and constant <= 255) :
              #if the constant is in image bounds use the constant, else use all zeros (black image)
              scaled = flatted_image
      else :
          #image not a costant ==> (tot_max - tot_min) != 0
          scaled = np.array([int(255 * (pixel - tot_min) / (tot_max - tot_min)) for pixel in flatted_image])
      # unflat image to original dimensions L*H
      unflatted = self.unflat_image(scaled)
      return unflatted
  

  # param image:         image as np.array
  # param filter_lambda: lambda : (x, y, pix) -> pix
  # param filter_size:   the size of filter
  # returns:             image as np.array with filter applied
  # NB  NO STRIDE is applied
  def apply_filter_as_lambda(self, image, filter_lambda, filter_size=(3,3)):
    im_h = image.shape[0]
    im_l = image.shape[1]
    # initialize output image to all zeros
    out_filtered_image = np.zeros(image.shape)
    # foreach pixel in image
    for im_x in range(-1, im_h-filter_size[0]+1):
      for im_y in range(-1, im_l-filter_size[1]+1):
        # foreach value in filter
        for fil_x in range(0, filter_size[0]):
          for fil_y in range(0, filter_size[1]):
              # input image indexes
              i = im_x + fil_x
              j = im_y + fil_y
              in_image_bounds = (i >= 0) and (j >= 0) and (i <= im_h) and (j <= im_l)
              if in_image_bounds :
                  out_filtered_image[im_x+1][im_y+1] += filter_lambda(fil_x, fil_y, image[i][j])
    # rescale pixels in [0, 255]
    return self.scale_image_to_int_in_bounds(out_filtered_image)


  # param image:  the original image 
  # param filter: the filter matrix (as np.array)
  # returns:      image as np.array with filter applied
  # NB  NO STRIDE is applied
  def apply_filter_as_matrix(self, image, filter_matrix) :
    # sizes definition
    filter_size = filter_matrix.shape                     #tipically (3,3) or (5,5)
    pad_size = (filter_size[0] // 2, filter_size[1] // 2) #tipically (1,1) or (2,2)
    # pad image to keep dimensionality
    padded_image = np.pad(image, pad_size)
    # initialize output image to list of empty rows
    out_filtered_image = [ [] for _ in range(self.im_h) ] 
    # foreach pixel in image
    for im_x in range(pad_size[0], self.im_h + pad_size[0]):
      for im_y in range(pad_size[1], self.im_l + pad_size[1]):
        new_pixel = 0
        # foreach value in filter
        for fil_x in range(0, filter_size[0]):
          for fil_y in range(0, filter_size[1]):
            # input image indexes
            im_i = im_x - pad_size[0] + fil_x
            im_j = im_y - pad_size[1] + fil_y
            new_pixel += padded_image[im_i][im_j] * filter_matrix[fil_x][fil_y]
        out_filtered_image[im_x - pad_size[0]].append(new_pixel)
    return np.array(out_filtered_image)
    

  # before use sobel apply horizontal and vertical filters:
  # param vertical_image:   apply_filter_as_matrix(image, horizontal_edge_filter)
  # param horizontal_image: apply_filter_as_matrix(image, vertical_edge_filter)
  # returns:                image as np.array with filter applied
  def apply_sobel_filter(self, vertical_image, horizontal_image) :
    # sobel = sqrt( ver^2 + hor^2 )
    # trasform pixels of vertical and horizontal images to their square
    vertical_image_square   = np.multiply(vertical_image,   vertical_image)
    horizontal_image_square = np.multiply(horizontal_image, horizontal_image)
    # return the square root of their sum
    return np.sqrt(vertical_image_square + horizontal_image_square)  


  # param image:  the original image
  # returns:      dictionary (Filter -> image as np.array with the filter applied)
  # NB the original image is not returned
  def generate_all_filters(self, image) :
    # dictionary: filter_name : filtered_image
    filtered_images = {}
    #---------------------------------------------------------------------------------
    # lambda filters : contrasts for now
    for filter_enum, filter_lambda in self.filters_lambda.items() :
      if filter_enum in self.filters_to_use :
            
        filtered_images[filter_enum] = self.apply_filter_as_lambda(image, filter_lambda)
    #---------------------------------------------------------------------------------
    # matrix filters : horizontal and vertical for now
    hor = ver = []
    for filter_enum, filter_matrix in self.filters_matrix.items() :
      # NB: sobel filter needs ver and hor calculations
      if (filter_enum in self.filters_to_use) or (Filters.SOBEL in self.filters_to_use) :
        # apply filter
        filtered_as_matrix = self.apply_filter_as_matrix(image, filter_matrix)
        # add filter only if it is requested
        if (filter_enum in self.filters_to_use) :
          filtered_images[filter_enum] = filtered_as_matrix
        # add it to the corresponding variable
        if filter_enum == Filters.HORIZONTAL :
            hor = filtered_as_matrix
        elif filter_enum == Filters.VERTICAL :
            ver = filtered_as_matrix
    #---------------------------------------------------------------------------------    
    # sobel filter
    if Filters.SOBEL in self.filters_to_use :
      filtered_images[Filters.SOBEL] = self.apply_sobel_filter(ver, hor)
    #---------------------------------------------------------------------------------    
    # horizontal flip
    if Filters.FLIP_HORIZONTAL in self.filters_to_use :
      filtered_images[Filters.FLIP_HORIZONTAL] = np.flip(image, 1)
    #---------------------------------------------------------------------------------    
    # rotations
    rot_min = -60
    rot_max = 60
    rot_step = 20
    filters_to_use_names = self.get_filters_names(self.filters_to_use)
    for degrees in range(rot_min, rot_max+1, rot_step) :
      direction = 'LEFT' if degrees < 0 else 'RIGHT'
      filter_name = 'ROT_' + direction + '_' + str(abs(degrees)) + '_DEGREES'
      if (degrees != 0) and (filter_name in filters_to_use_names) :
        filter_enum = Filters[filter_name]
        filtered_images[filter_enum] = imutils.rotate(image, degrees)
    # return all filters
    return filtered_images
  
  
  # param filters: list of filters as enum
  # returns:       list of filters as string
  def get_filters_names(self, filters) :
    return [ e.name for e in filters ]
  

  # returns:  a list of lists of string: foreach image a list with the original image and all filters applied
  def build_new_dataset(self) :
    new_dataset = []
    print("converting pixels from string to nparray...")
    np_images = self.string_to_array(df['pixels'])
    print("generating filters for all images...")
    i = 0
    for original_image in tqdm(np_images):
      # image list
      all_image_filters = []
      if Filters.ORIGINAL in self.filters_to_use :
        # add original image to image list
        self.filters_used_ordered = [ Filters.ORIGINAL ]
        all_image_filters.append(df.at[i, 'pixels'])
      # generate all filters of image
      filtered_images = self.generate_all_filters(original_image)
      # add one by one all the filters to image list
      for filter_enum in filtered_images :
        cur_filtered_image = filtered_images[filter_enum]
        self.filters_used_ordered.append(filter_enum)
        all_image_filters.append(self.array_to_string(self.flat_image(cur_filtered_image)))
      # add all the image list to dataset
      new_dataset.append(all_image_filters)
      i += 1
    return new_dataset
  
  
  # returns: the dataset with the filters applied as a df
  def create_new_df(self) :
    new_dataset = self.build_new_dataset()
    new_columns = self.get_filters_names(self.filters_used_ordered)
    new_dataset_df = pd.DataFrame(new_dataset, columns = new_columns)    
    emotions_usage_df = self.data[['emotion','Usage']]
    return pd.concat([emotions_usage_df, new_dataset_df], axis=1)  
    
    
  # param df_output: dataframe with filters produced by create_new_df()
  # param rand_seed: seed to use for index of image to generate (default random)
  # param cols:      number of pictures to print in a row
  # param figsize:   size of the pictures
  # prints image example with filters used (needed a notebook to see the output)
  def print_example(self, df_output, rand_seed=None, cols=4, figsize=(10,10)) :
    import matplotlib.pyplot as plot
    from random import seed
    from random import random
    # set random seed
    seed(rand_seed)
    # obtain random index
    img_idx = int(random() * len(self.data))
    # print index and emotion
    # print image with all filters
    print('img_idx:', img_idx)
    print('emotion:', self.emotions[self.data['emotion'][img_idx]])
    num_images = len(self.filters_used_ordered)
    # calculate number of columns needed
    rows = math.ceil(num_images / cols)
    # create boxes
    fig, axs = plot.subplots(rows, cols, figsize=figsize)
    i = j = cur_im = 0
    filter_columns = self.get_filters_names(self.filters_used_ordered)
    row = df_output[img_idx:(img_idx+1)]   
    for column in row :
        if column in filter_columns :
            image = self.string_to_array_image(row[column][img_idx])
            idxs = max(i,j) if (rows == 1 or cols == 1) else (i,j)
            subpl=axs[idxs]
            subpl.imshow(image, cmap='gray')
            subpl.set_title(column)
            plot.tight_layout()
            plot.subplots_adjust(hspace=None)
            cur_im += 1
            j = (j + 1) % cols
            if j == 0 :
              i += 1
    # remove empty boxes
    for r in range(rows) :
        for c in range(cols) :
            if (c + r * cols) >= num_images :
                fig.delaxes(axs[r][c])
          
    
    
    
# ------------------------------------------------------------------------------------------------
# main    

dataset = 'fer2013.csv'
df = pd.read_csv(dataset)[:3]

y = df['emotion']

#filters = [ e for e in Filters ]
filters = [ Filters.ORIGINAL, Filters.SOBEL, Filters.VERTICAL, Filters.CONTRAST_HIGH, Filters.CONTRAST_LOW, Filters.CONTRAST_VERY_HIGH, Filters.ROT_LEFT_40_DEGREES, Filters.ROT_LEFT_20_DEGREES ]
filter_names = [ f.name for f in filters ]

my_class = Fer2013_Augmenter(df, filters)

out_df = my_class.create_new_df()

my_class.print_example(out_df)
