# IMPORT LIBS
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn; cudnn.benchmark = True

# DEFINE MODELS
class LSTM(tsd):
  def __init__(self, input_size, lstm_size, lstm_layers, output_size):
    # Call parent
    super(Model, self).__init__()
    self.input_size = input_size
    self.lstm_size = lstm_size
    self.lstm_layers = lstm_layers
    self.output_size = output_size
    # Define internal modules (LSTM)
    self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
    
    self.output = nn.Linear(lstm_size, output_size)
    
  def forward(self, x):
    # Prepare LSTM initial state
    proposal_num = x.size(0)
    lstm_init = (torch.zeros(self.lstm_layers, proposal_num, self.lstm_size), torch.zeros(self.lstm_layers, proposal_num, self.lstm_size))
    if x.is_cuda:
      lstm_init = (Variable(lstm_init[0].cuda(), volatile=x.volatile), Variable(lstm_init[1].cuda(), volatile=x.volatile))
    else:
      print("CUDA is not implemented")
    # Forward LSTM and get final state
    x = self.lstm(x, lstm_init)[0][:,-1,:]
    x = self.output(x)
    
    return x
  
def Optimizer(opt, model):
  optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr = opt.lstm_learning_rate)
  return optimizer



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

class _LSTM(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    # Call parent
    super(LSTM, self).__init__()

    self.input_size = input_size
    self.lstm_size = lstm_size
    self.lstm_layers = lstm_layers
    self.output_size = output_size
    # Define internal modules (LSTM)
    self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
    
    self.output = nn.Linear(lstm_size, output_size)
       
  def forward(self, x):
    # Prepare LSTM initial state
    proposal_num = x.size(0)
    lstm_init = (torch.zeros(self.lstm_layers, proposal_num, self.lstm_size), torch.zeros(self.lstm_layers, proposal_num, self.lstm_size))
    if x.is_cuda:
      lstm_init = (Variable(lstm_init[0].cuda(), volatile=x.volatile), Variable(lstm_init[1].cuda(), volatile=x.volatile))
    else:
      print("CUDA is not implemented")
    # Forward LSTM and get final state
    x = self.lstm(x, lstm_init)[0][:,-1,:]
    x = self.output(x)
    
    return x

class resnet(_tsd):
  def __init__(self, classes, num_layers=1, pretrained=False, class_agnostic=False):
    #self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    self.dout_base_model = 1024
    self.pretrained = 0
    self.class_agnostic = class_agnostic

    _tsd.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    LSTM = _LSTM()

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet.
    self.RCNN_base = nn.Sequential(_LSTM.lstm) # Extract the first output from LSTM network

    self.RCNN_cls_score = nn.Linear(1024, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(1024, 2)
    else:
      self.RCNN_bbox_pred = nn.Linear(1024, 2 * self.n_classes)
