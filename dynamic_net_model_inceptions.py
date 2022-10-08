import torch
import torch.nn as nn

class DynamicNetInceptions(nn.Module):
    
    def __init__(self, device, \
      drop__before_relu, drop__before_incep, drop__before_linear, \
      conv__layer_repetitions, conv__in_channels, conv__out_channels, \
      lin__out_dimension, \
      incep__num_layers, incep__multiplier, im_h, im_l) :
      
      super(DynamicNetInceptions, self).__init__()

      self.device = device

      # PARAMETERS SECTION
      self.pool_size = (2,2)
      self.kernel_size = (3,3)
      self.padding = 1
      
      self.num_of_conv_layers = len(conv__layer_repetitions)
      self.num_of_lin_layers = len(lin__out_dimension)
      
      self.drop__before_incep  = drop__before_incep
      self.drop__before_linear = drop__before_linear
      
      self.incep__num_layers = incep__num_layers
      self.incep__multiplier = incep__multiplier
             
      in_chan = conv__in_channels
      self.convs = []

      if (len(conv__layer_repetitions) != len(conv__out_channels)) :
        print("Error: len(conv__layer_repetitions) != len(conv__out_channels)")
        
      self.skip = nn.Identity()

      # CONVOLUTIONAL SECTION
      for big_layer in range(self.num_of_conv_layers) :
        out_chan = conv__out_channels[big_layer]
        cur_rep_big_layer = conv__layer_repetitions[big_layer]
        for repeat_layer in range(cur_rep_big_layer) : #a, b, c, ...
          self.convs.append(nn.Conv2d(in_chan, out_chan, self.kernel_size, padding=self.padding))
          if drop__before_relu > 0 :
            self.convs.append(nn.Dropout2d(p=drop__before_relu))
          self.convs.append(nn.ReLU())
          in_chan = out_chan
        self.convs.append(nn.MaxPool2d(self.pool_size))

      self.drop__before_incep = nn.Dropout2d(p=drop__before_incep).to(self.device)

      # INCEPTION SECTION
      if (incep__num_layers > 0) :
        in_chan = out_chan
        self.incep__multiplier = incep__multiplier
        self.incep__num_layers = incep__num_layers
        self.incep_first = self.inception_module(in_chan)
        self.incep_after = []
        total_out_chan = incep__multiplier * (64 + 128 + 32 + 32) # 256 * out_multiplier
        in_chan = total_out_chan
        for inception_layer in range(incep__num_layers - 1) :
            cur_incep = self.inception_module(in_chan)
            self.incep_after += [ cur_incep ]
        
      self.drop__before_linear = nn.Dropout2d(p=drop__before_linear).to(self.device)

      #final output dimension
      self.final_conv_dim = total_out_chan * (im_h // (2 ** self.num_of_conv_layers)) * (im_l // (2 ** self.num_of_conv_layers))

      # LINEAR FC SECTION
      in_dim = self.final_conv_dim

      self.linear_fc = []
      for lin_layer in range(self.num_of_lin_layers) :
        out_dim = lin__out_dimension[lin_layer]
        self.linear_fc.append(nn.Linear(in_dim, out_dim))
        self.linear_fc.append(nn.ReLU())
        in_dim = out_dim
      
      # SEQUENTIAL SECTION
      self.convs_seq     = nn.Sequential(*self.convs).to(self.device)
      self.linear_fc_seq = nn.Sequential(*self.linear_fc).to(self.device)
      self.softmax       = nn.Softmax(1).to(self.device)

    def inception_module(self, in_chan, out_1x1=64, out_3x3=[96,128], out_5x5=[16,32], out_pool=32) :
      
      mul = self.incep__multiplier
    
      branch_1x1 = nn.Sequential(
        nn.Conv2d(in_chan, out_1x1*mul, kernel_size=1) #conv 1x1
      ).to(self.device)

      branch_3x3 = nn.Sequential(
        nn.Conv2d(in_chan,          (out_3x3[0])*mul, kernel_size=1),           # conv 1x1
        nn.Conv2d((out_3x3[0])*mul, (out_3x3[1])*mul, kernel_size=3, padding=1) # conv 3x3
      ).to(self.device)

      branch_5x5 = nn.Sequential(
        nn.Conv2d(in_chan,          (out_5x5[0])*mul, kernel_size=1),           # conv 1x1
        nn.Conv2d((out_5x5[0])*mul, (out_5x5[1])*mul, kernel_size=5, padding=2) # conv 5x5
      ).to(self.device)

      branch_pool = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=0), # max_pool 3x3
        nn.Conv2d(in_chan, out_pool*mul, kernel_size=1, stride=1, padding=1) # conv 1x1
      ).to(self.device)
        
      return [ branch_1x1, branch_3x3, branch_5x5, branch_pool ]

    def print_net(self) :
        
      print("__Convolutionals Start__")
      print(self.convs_seq)
      print("__Convolutionals End__")
      print()
      
      if self.drop__before_incep > 0 : 
        print(self.drop__before_incep)
        print()
    
      if self.incep__num_layers > 0 :
        n = self.incep__num_layers
        mul = self.incep__multiplier
        print("__Inception Start (with skip)__")
        print(f"Inception__1 with dim: N -> (256 * {mul})")
        print(self.incep_first)
        print(f"Inception__2,...,{n} with dim: (256 * {mul}) -> (256 * {mul})")
        print(self.incep_after[0])
        print("__Inception End__")
        print()
        
      if self.drop__before_linear > 0 :
        print(self.drop__before_linear)
        print()
    
      print(f"Reshape(-1, {self.final_conv_dim})")
      print()
    
      print("__Linear Start__")
      print(self.linear_fc_seq)
      print("__Linear End__")
      print()
    
      print(self.softmax)
      print()

    def run_inception(self, x, inception) :
      x_1x1  = (inception[0])(x)
      x_3x3  = (inception[1])(x)
      x_5x5  = (inception[2])(x)
      x_pool = (inception[3])(x)
      concat = torch.cat((x_1x1, x_3x3, x_5x5, x_pool), 1)
      return concat
    
    def forward(self, x):

      # convolutional part
      x = self.convs_seq(x)
    
      # dropout
      if self.drop__before_incep > 0 : 
        x = self.drop__before_incep(x)
        
      # inceptions
      if self.incep__num_layers > 0 :
        x = self.run_inception(x, self.incep_first) 
        for inception in self.incep_after : 
          x = self.run_inception(x, inception) + self.skip(x)

      # dropout
      if self.drop__before_linear > 0 : 
        x = self.drop__before_linear(x)
    
      # reshape
      x = x.reshape(-1, self.final_conv_dim)
      
      # linear
      x = self.linear_fc_seq(x)
      
      #softmax
      x = self.softmax(x)

      return x
