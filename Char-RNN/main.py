import dataset
from model import CharRNN, CharLSTM
from tqdm import tqdm 
import torch 
import time
import string
from torch import nn 
from torch.utils import data as da 
from torch.utils.data.sampler import SubsetRandomSampler

# import some packages you need here


def train(model, trn_loader, device, criterion, optimizer, epoch, vocab_size, mode="RNN"):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """
    trn_loss = 0
    hidden = model.init_hidden(1)
    start_time = time.time()
    trn_loss = 0
    for i, (data) in enumerate(trn_loader) :
        model.train()
        x = data[:, :-1]
        y = data[:, 1:]

        x = x.to(device)
        y = y.to(device) 

        y_pred, _  = model(x)
        loss = criterion(y_pred.view(-1, vocab_size), y.view(-1))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        trn_loss += (loss)
        end_time = time.time()
        print(" [Training] [{0}] [{1}/{2}] Losses = [{3:.4f}] Time(Seconds) = [{4:.2f}]".format(epoch, i, len(trn_loader), loss.item(), end_time-start_time))
        start_time = time.time()
        '''     
        if use only shakepears datasets2 
        model.zero_grad()
        loss = 0 

        for z in range(len(input)) :
           # import ipdb; ipdb.set_trace() 
            y_pred, hidden = model(input[z].cuda(), hidden.cuda())
            loss += criterion(y_pred, target[z].cuda())


        loss.backward(retain_graph=True)
        optimizer.step()
        end_time = time.time()
        
        print("[{0}] [{1}/{2}] Losses = [{3:.4f}] Time(Seconds) = [{4:.2f}]".format(epoch, i, 100, loss.item(), end_time-start_time))
        trn_loss += loss.item()
        start_time = time.time()

        if i == 100 :
            break '''
    # write your codes here

    trn_loss = trn_loss/len(trn_loader)
    if epoch == 50 : 
        torch.save(model.state_dict(), '{0}{1}_{2}_{3}.pth'.format("./", 'model', epoch, mode))

    return trn_loss

def validate(model, val_loader, device, criterion, epoch, vocab_size):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """
    model.eval()
    val_loss = 0 
    start_time = time.time()
    with torch.no_grad() :
        for i, (data) in enumerate(val_loader) :
            model.train()
            x = data[:, :-1]
            y = data[:, 1:]

            x = x.to(device)
            y = y.to(device) 

            y_pred, _  = model(x)
            loss = criterion(y_pred.view(-1, vocab_size), y.view(-1))
            val_loss += (loss)
            end_time = time.time()
            print(" [Validation] [{0}] [{1}/{2}] Losses = [{3:.4f}] Time(Seconds) = [{4:.2f}]".format(epoch, i, len(val_loader), loss.item(), end_time-start_time))
            start_time = time.time()

    # write your codes here
    val_loss = val_loss / len(val_loader)

    return val_loss

def draw_curve(logger_train, logger_test, model="RNN") :
    import matplotlib.pyplot as plt 
    import seaborn as sns 

    fig = plt.gcf()
    plt.plot(logger_train, c='purple', label = "Training Loss")
    plt.plot(logger_test, c='skyblue', label = "Test Loss")
    plt.title("Compare Loss, {0}".format(model))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig("{0}_Loss.png".format(model))
    plt.close()

def main(mode="RNN"):
    """
     Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
            Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss
    """

    # write your codes here
    start_time = time.time()
    data = "./shakespeare_train.txt"
    data_set = dataset.Shakespeare(data)
    all_characters = string.printable
    input_size = len(all_characters)

    if mode == "RNN" :
        models_RNN = CharRNN(input_size, 512, input_size, 4).cuda()
        optim = torch.optim.Adam(models_RNN.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
    elif mode == "LSTM" :
        models_LSTM = CharLSTM(input_size, 512, input_size, 4).cuda()
        optim = torch.optim.Adam(models_LSTM.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
    else :
        raise NotImplementedError

    all_losses = [] 
    loss_avg = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #sampler = SubsetRandomSampler()
    total_idx = list(range(len(data_set)))
    split_idx = int(len(data_set) * 0.7)
    trn_idx = total_idx[:split_idx]
    val_idx = total_idx[split_idx:]

    trn_loader = da.DataLoader(data_set, batch_size=64, sampler=SubsetRandomSampler(trn_idx))
    val_loader = da.DataLoader(data_set, batch_size=64, sampler=SubsetRandomSampler(val_idx))
    losses = []
    val_losses = []
    for epoch in range(1, 51) :
        if mode == "RNN" : 
            loss = train(models_RNN, trn_loader, device, criterion, optim, epoch, input_size, mode="RNN")
            val_loss = validate(models_RNN, val_loader, device, criterion, epoch, input_size)
        elif mode == "LSTM" :
            loss = train(models_LSTM, trn_loader, device, criterion, optim, epoch, input_size, mode="LSTM")
            val_loss = validate(models_LSTM, val_loader, device, criterion, epoch, input_size)
        losses.append(loss)
        val_losses.append(val_loss)
    
    return losses, val_losses


if __name__ == '__main__':
    trn_loss, val_loss = main(mode="RNN")
    LSTM_trn_loss, LSTM_val_loss = main(mode="LSTM")
    draw_curve(trn_loss, val_loss, model="RNN")
    draw_curve(LSTM_trn_loss, LSTM_val_loss, model="LSTM")

    import ipdb; ipdb.set_trace()
    