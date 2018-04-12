# IMPORT LIBS
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn; cudnn.benchmark = True

# DEFINE MODELS
class Model(nn.Module):
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
