import os
import dataset
from model import LeNet5, CustomMLP, GeunSungCustomMLP
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import tarfile, os, shutil


def train(model, trn_loader, device, criterion, optimizer, epoch):
    trn_tmp = 0
    acc_tmp = 0
    count_data = 0
    for iter, (i, j) in enumerate(trn_loader) :
        images, labels = i.to(device), j.to(device)
        result = model(images).to(device)
        loss = criterion(result.to(device), labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trn_tmp += loss.item()
        count_data += len(i)
        _, predicted = torch.max(result, 1)
        tmp_acc = (predicted==labels).sum().item()
        acc_tmp += (predicted == labels).sum().item()
        tmp_acc = tmp_acc / len(i)
        length = len(trn_loader)
        print("Epoch : [{1}] [{0}/{4}], Loss : [{2:3.4f}], Acc : [{3:3.3f}]".format(iter, epoch, loss.item(), tmp_acc, length))


    trn_loss = trn_tmp / len(trn_loader)
    acc = acc_tmp / count_data

    print("[{0}] Training, Loss : [{1:3.4f}], Acc : [{2:3.3f}]".format(epoch, trn_loss, acc))

    return trn_loss, acc

def test(model, tst_loader, device, criterion, epoch):
    tst_tmp = 0
    acc_tmp = 0
    data_count = 0
    with torch.no_grad() :
        for iter, (image, labels) in enumerate(tst_loader) :
            image, labels = image.to(device), labels.to(device)
            result = model(image).to(device)
            loss = criterion(result, labels)
            tst_tmp += loss.item()
            _, predict = torch.max(result, 1)
            acc_tmp += (predict == labels).sum().item()
            tmp_acc = (predict == labels).sum().item()
            tmp_acc = tmp_acc / len(image)
            data_count += len(image)
            length = len(tst_loader)
            print("Epoch : [{1}] [{0}/{4}] Validation, Loss : [{2:3.4f}], Acc : [{3:3.3f}]".format(iter, epoch, loss.item(), tmp_acc, length))

        tst_loss = tst_tmp / len(tst_loader)
        acc = acc_tmp / data_count

    print("[{0}] Test, Loss : [{1:3.4f}], Acc : [{2:3.3f}]".format(epoch, tst_loss, acc))

    return tst_loss, acc


def main(net="lenet5", augmentation=False):
    device = torch.device("cpu")
    if net == "lenet5" :
        net = LeNet5()
    elif net == "mlp" :
        net = CustomMLP()
    else :
        print("Not support Networks")
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    if os.path.isdir("../data/train_mnist") == True :
        pass
    else :
        print("Unzip train mnist")
        trn_dataset = '../data/train.tar'
        trn_dst_path = '../data/train_mnist'
        os.makedirs(trn_dst_path)
        with tarfile.TarFile(trn_dataset, 'r') as file:
            file.extractall(trn_dst_path)

    if os.path.isdir("../data/test_mnist") == True :
        pass
    else :
        print("Unzip test mnist")
        tst_dataset = '../data/test.tar'
        tst_dst_path = '../data/test_mnist'
        os.makedirs(tst_dst_path)
        with tarfile.TarFile(tst_dataset, 'r') as file:
            file.extractall(tst_dst_path)

    train_folder = "../data/train_mnist/train"
    test_folder = "../data/test_mnist/test"

    train_set = dataset.MNIST(train_folder)
    test_set = dataset.MNIST(test_folder)
    if augmentation == True :
        aug_train_set = dataset.MNIST(train_folder, aug=True)
        aug_test_Set = dataset.MNIST(test_folder)

    trn_loader = DataLoader(train_set, batch_size=128)
    tst_loader = DataLoader(test_set, batch_size=128)
    if augmentation == True :
        aug_train_loader = DataLoader(aug_train_set, batch_size=128)
        aug_test_loader = DataLoader(aug_test_Set, batch_size=128)

    train_logger = []
    test_logger = []

    for epoch in range(5) :
        if augmentation == False :
            training = train(net.to(device), trn_loader, device, loss, optimizer, epoch)
            tests = test(net.to(device), tst_loader, device, loss, epoch)
            train_logger.append(training)
            test_logger.append(tests)
        else :
            training = train(net.to(device), aug_train_loader, device, loss, optimizer, epoch)
            tests = test(net.to(device), tst_loader, device, loss, epoch)
            train_logger.append(training)
            test_logger.append(tests)

    return train_logger, test_logger

def draw_curve(logger_train, logger_test, model="LeNet") :
    losses_train = []
    acces_train = []
    losses_test = []
    acces_test = []
    for (train_loss, train_acc), (test_loss, test_acc) in zip(logger_train, logger_test) :
        losses_train.append(train_loss)
        losses_test.append(test_loss)
        acces_train.append(train_acc)
        acces_test.append(test_acc)
    import matplotlib.pyplot as plt

    fig = plt.gcf()
    plt.plot(losses_train, c='purple', label = "Training Loss")
    plt.plot(losses_test, c='skyblue', label = "Test Loss")
    plt.title("Compare Loss, {0}".format(model))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("{0}_Loss.png".format(model))
    plt.close()
    #fig.close()

    fig = plt.gcf()
    plt.plot(acces_train, c='purple', label = "Training acc")
    plt.plot(acces_test, c='skyblue', label = "Test acc")
    plt.title("Compare ACC, {0}".format(model))
    plt.xlabel("Epoch")
    plt.ylabel("ACC")
    plt.legend()
    plt.show()
    plt.savefig("{0}_ACC.png".format(model))
    plt.close()
    return losses_train, losses_test, acces_train, acces_test



if __name__ == '__main__':
    train_log_mlp, test_log_mlp = main("mlp")
    train_log_lenet, test_log_lenet = main("lenet5")

    draw_curve(train_log_mlp, test_log_mlp, "Multi Layer Perceptron")
    draw_curve(train_log_lenet, test_log_lenet, "LeNet5")

    train_log_mlp_aug, test_log_mlp_aug = main("mlp", augmentation=True)
    train_log_lenet_aug, test_log_lenet_aug = main("lenet5", augmentation=True)

    draw_curve(train_log_mlp_aug, test_log_mlp_aug, "Multi Layer Perceptron With Augmentation")
    draw_curve(train_log_lenet_aug, test_log_lenet_aug, "LeNet5 With Augmentation")
    import ipdb; ipdb.set_trace()
