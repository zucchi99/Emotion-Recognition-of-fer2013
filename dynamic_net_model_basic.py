import torch
import torch.nn as nn

class DynamicNetBasic(nn.Module):
    
    def __init__(self, device, dropout_prob, conv__layer_repetitions, conv__in_channels, conv__out_channels, lin__out_dimension, im_h, im_l) :
      
      super(DynamicNetBasic, self).__init__()

      self.device = device
      
      #params
      self.pool_size = (2,2)
      self.kernel_size = (3,3)
      self.padding = 1
      self.num_of_conv_layers = len(conv__layer_repetitions)
      self.num_of_lin_layers = len(lin__out_dimension)
      self.dropout_prob = dropout_prob
      
      in_chan = conv__in_channels
      self.convs = [] 

      if (len(conv__layer_repetitions) != len(conv__out_channels)) :
        print("ERROR")

      for big_layer in range(self.num_of_conv_layers) :
        out_chan = conv__out_channels[big_layer]
        cur_rep_big_layer = conv__layer_repetitions[big_layer]
        for repeat_layer in range(cur_rep_big_layer) : #a, b, c, ...
          self.convs += [ nn.Conv2d(in_chan, out_chan, self.kernel_size, padding=self.padding) ]
          #self.convs += [ nn.Dropout2d(p=self.dropout_prob) ]
          self.convs += [ nn.ReLU() ]
          in_chan = out_chan
        self.convs += [nn.MaxPool2d(self.pool_size)]

      self.convs += [ nn.Dropout2d(p=self.dropout_prob) ]
      
      #final output dimension
      self.final_conv_dim = out_chan * (im_h // (2 ** self.num_of_conv_layers)) * (im_l // (2 ** self.num_of_conv_layers))

      # linears
      in_dim = self.final_conv_dim

      self.linear_fc = []
      for lin_layer in range(self.num_of_lin_layers) :
        out_dim = lin__out_dimension[lin_layer]
        self.linear_fc += [ nn.Linear(in_dim, out_dim) ]
        #self.linear_fc += [ nn.Dropout2d(p=self.dropout_prob) ]
        self.linear_fc += [ nn.ReLU() ]
        in_dim = out_dim
      
      self.softmax = nn.Softmax(1)

      self.features = nn.Sequential(*self.convs).to(self.device)
      self.linear_fc = nn.Sequential(*self.linear_fc).to(self.device)

      #print(self.features)
      #print(self.linear_fc)

    def print_net(self) :
      print(self.features)
      print(f"Reshape(-1, {self.final_conv_dim})")
      print(self.linear_fc)
      print(self.softmax)

    def forward(self, x):
      
      x = self.features(x)
      x = x.reshape(-1, self.final_conv_dim)
      x = self.linear_fc(x)
      
      x = self.softmax(x)

      return x

#net = DynamicNetBasic(0.1,conv__layer_repetitions=(2,2,1), conv__in_channels=3)
#net.print_net()