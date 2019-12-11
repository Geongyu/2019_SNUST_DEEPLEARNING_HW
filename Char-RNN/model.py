import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np 
from torch.utils import data as da 
import dataset
from tqdm import tqdm 
import time
import string
from torch.utils import data as da 
from torch.utils.data.sampler import SubsetRandomSampler

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        # write your codes here
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn_layer = nn.RNN(hidden_size, hidden_size, n_layers, dropout=0.5, batch_first=True)
        self.decode = nn.Linear(hidden_size, input_size)

    def forward(self, input, hidden=None):
        # write your codes here
       # import ipdb; ipdb.set_trace()
        x = self.embedding(input)
        output, hidden = self.rnn_layer(x, hidden)
        output = self.decode(output)

        return output, hidden

    def init_hidden(self, batch_size):

        # write your codes here
        initial_hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        return initial_hidden

class CharRNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        # write your codes here
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn_layer = nn.RNN(hidden_size, hidden_size, output_size)
        self.decode = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden=None):
        # write your codes here
       # import ipdb; ipdb.set_trace()
        x = self.embedding(input.view(1, -1))
        output, hidden = self.rnn_layer(x.view(1, 1, -1), hidden)
        output = self.decode(output.view(1, -1))

        return output, hidden

    def init_hidden(self, batch_size):

        # write your codes here
        initial_hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        return initial_hidden


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1) :
        super(CharLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=0.5)
        self.decode = nn.Linear(hidden_size, input_size)

        # write your codes here

    def forward(self, input, hidden=None ):
        x = self.embedding(input)
        output, hidden = self.lstm(x, hidden)
        output = self.decode(output)
        return output, hidden

    def init_hidden(self, batch_size):

        # write your codes here
        initial_hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        return initial_hidden

class CharLSTM2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1) :
        super(CharLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.2)
        self.decode = nn.Linear(hidden_size, output_size)

        # write your codes here

    def forward(self, input, hidden=None):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):

        # write your codes here
        initial_hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        return initial_hidden

if __name__ == '__main__':
    ds = dataset.Shakespeare("./shakespeare_train.txt", chuck_size=30)
    loader = da.DataLoader(ds, batch_size=1, shuffle=False)
    all_chars = string.printable
    aa = CharRNN(100, 100, 100, 3)
    hidden = aa.init_hidden(1)
    for i, j in loader :
        aa(i, hidden)