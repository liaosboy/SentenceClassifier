import torch
import torch.nn as nn
from torch.autograd import Variable

class Model(nn.Module):
  def __init__(self, input_size, seq_size, hidden_size, num_layers, num_class):
    super(Model,self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.seq_size = seq_size
    self.normal = nn.BatchNorm1d(seq_size, affine=True)
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size*seq_size, num_class)
    self.softmax = nn.LogSoftmax()

  def forward(self, x):
    h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
    c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
    out = self.normal(x)
    out, _ = self.lstm(out, (h0, c0))
    out = out.reshape(-1, self.hidden_size * self.seq_size)
    out = self.fc(out)
    out = self.softmax(out)
    return out