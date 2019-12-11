# import some packages you need here
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchtext
import string 

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
                        You need this dictionary to generate characters.
                2) Make list of character indices using the dictionary
                3) Split the data into chunks of sequence length 30. 
            You should create targets appropriately.
    """

    def __init__(self, input_file, chuck_size=30):
        # write your codes here
        all_chars = string.printable
        vacab_size = len(all_chars)
        vocab_dict = dict((c, i) for (i, c) in enumerate(all_chars))

        data = str2int(open(input_file).read().strip(), vocab_dict)

        data = torch.tensor(data, dtype=torch.int64).split(chuck_size)

        if len(data[-1]) < chuck_size :
            data = data[:-1] 
        self.data = data
        self.n_chunck = len(self.data)        

    def __len__(self):
        # write your codes here
        return self.n_chunck

    def __getitem__(self, idx):
        # write your codes here
        return self.data[idx]
        '''target = self.text[idx+1:idx+self.chunk_size]
        return input, target'''

class Shakespeare2(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
                        You need this dictionary to generate characters.
                2) Make list of character indices using the dictionary
                3) Split the data into chunks of sequence length 30. 
            You should create targets appropriately.
    """

    def __init__(self, input_file, chuck_size=30):
        # write your codes here
        all_chars = string.printable
        vacab_size = len(all_chars)
        self.vocab_dict = dict((c, i) for (i, c) in enumerate(all_chars))
        self.inputfile = open(input_file).read()
        self.chunk_size = chuck_size
        self.text = str2int(open(input_file).read().strip(), self.vocab_dict)
        

    def __len__(self):
        # write your codes here
        return len(self.inputfile) - self.chunk_size

    def __getitem__(self, idx):
        # write your codes here
        input = self.text[idx:idx + self.chunk_size-1]
        #return input
        target = self.text[idx+1:idx+self.chunk_size]
        return input, target
    
    def return_vocab_dict(self) :
        return self.vocab_dict

def str2int(s, vocab_dict) :
    return [vocab_dict[c] for c in s]

def int2str(x, vocab_array) :
    return "".join([vocab_array[i] for i in x])

if __name__ == '__main__':
    ds = Shakespeare("./shakespeare_train.txt", chuck_size=30)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    all_chars = string.printable
    for i, j in loader :
        
        import ipdb; ipdb.set_trace()
        vocab = ds.return_vocab_dict()
        print(int2str(j.item(), vocab))

    # write test codes to verify your implementations