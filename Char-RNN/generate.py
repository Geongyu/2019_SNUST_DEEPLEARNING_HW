# import some packages you need here
import torch
import numpy as np 
import pandas as pd 
import string 
from model import CharRNN, CharLSTM

def str2int(s, vocab_dict) :
    return [vocab_dict[c] for c in s]

def int2str(x, vocab_array) :
    return "".join([vocab_array[i] for i in x])

def generate(model, seed_characters, temperature, device, length=30, *args):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
                temperature: T
                args: other arguments if needed

    Returns:
        samples: generated characters
    """
    all_chars = string.printable
    vacab_size = len(all_chars)
    vocab_dict = dict((c, i) for (i, c) in enumerate(all_chars))

    model.eval()
    result = []

    start_tensor = torch.tensor(str2int(seed_characters, vocab_dict), dtype=torch.int64).to(device)

    x0 = start_tensor.unsqueeze(0) 
    o, h = model(x0)
    out_dist = o[:, -1].view(-1).exp()
    top_i = torch.multinomial(out_dist, 1)[0]
    result.append(top_i)

    for i in range(length) :
        inp = torch.tensor([[top_i]], dtype=torch.int64)
        inp = inp.to(device)
        o, h = model(inp, h)
        out_dist = o.view(-1).div(temperature).view(-1).exp().cpu()
        top_i = torch.multinomial(out_dist, 1)[0]
        result.append(top_i)

    return seed_characters + int2str(result, all_chars)

if __name__ == '__main__':
    all_characters = string.printable
    input_size = len(all_characters)

    pre = torch.load("./model_50_RNN.pth")
    models_RNN = CharRNN(input_size, 512, input_size, 4).cuda()

    models_RNN.load_state_dict(pre)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    temp = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rnns = []
    for i in temp :
        result = generate(models_RNN, "A", i, device)
        rnns.append(result)
    
    with open("RNN_Result_50.txt", "w") as f :
        for item in rnns :
            f.write("%s\n" %item)
    f.close()

    pre = torch.load("./model_50_LSTM.pth")
    models_LSTM = CharLSTM(input_size, 512, input_size, 4).cuda()

    models_LSTM.load_state_dict(pre)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    temp = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    lstms = []
    for i in temp :
        result = generate(models_LSTM, "A", i, device)
        lstms.append(result)
    

    with open("LSTM_Result_50.txt", "w") as f :
        for item in lstms :
            f.write("%s\n" %item)

    import ipdb; ipdb.set_trace()