import argparse
import utils
import random
import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from models import CNN, BiLSTM
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(model.parameters(), args.learning_rate)

    def train(self, data_loader):
        self.model.train()

        loss_list = []
        pred_list = []
        label_list = []
        for labels, inputs in data_loader:
            self.optimizer.zero_grad()

            outputs = self.model(inputs)


            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            pred_list.append(torch.argmax(outputs, dim=-1).cpu().numpy())
            label_list.append(labels.cpu().numpy())

        y_pred = np.concatenate(pred_list)
        y_true = np.concatenate(label_list)

        loss = np.mean(loss_list)
        acc = accuracy_score(y_true, y_pred)

        return loss, acc

    def evaluate(self, data_loader):
        self.model.eval()

        loss_list = []
        pred_list = []
        label_list = []
        with torch.no_grad():
            for labels, inputs in data_loader:
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                loss_list.append(loss.item())
                pred_list.append(torch.argmax(outputs, dim=-1).cpu().numpy())
                label_list.append(labels.cpu().numpy())

        y_pred = np.concatenate(pred_list)
        y_true = np.concatenate(label_list)

        loss = np.mean(loss_list)
        acc = accuracy_score(y_true, y_pred)

        return loss, acc

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', type=str, default='./data/train.csv')
    parser.add_argument('-dev_file', type=str, default='./data/dev.csv')
    parser.add_argument('-test_file', type=str, default='./data/test.csv')
    parser.add_argument('-save_path', type=str, default='./model.pkl')
    parser.add_argument('-model', type=str, default="bilstm", help="[cnn, bilstm]")

    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-embedding_size', type=int, default=128)
    parser.add_argument('-hidden_size', type=int, default=64)
    parser.add_argument('-learning_rate', type=float, default=5e-4)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-epochs', type=int, default=20)

    parser.add_argument('-seed', type=int, default=1)
    args = parser.parse_args()

    print(args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("Loading Data...")
    dataests, vocab = utils.build_dataset(args.train_file,
                                          args.dev_file,
                                          args.test_file)

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=args.batch_size,
                   collate_fn=utils.collate_fn,
                   shuffle=dataset.train)
        for i, dataset in enumerate(dataests))

    print("Building Model...")
    if args.model == "cnn":
        model = CNN.Model(vocab_size=len(vocab),
                          embedding_size=args.embedding_size,
                          hidden_size=args.hidden_size,
                          filter_sizes=[3, 4, 5],
                          dropout=args.dropout)
    elif args.model == "bilstm":
        model = BiLSTM.Model(vocab_size=len(vocab),
                             embedding_size=args.embedding_size,
                             hidden_size=args.hidden_size,
                             dropout=args.dropout)

    if torch.cuda.is_available():
        model = model.cuda()

    trainer = Trainer(model)
    tloss=[]
    tacc=[]
    dloss=[]
    dacc=[]
    best_acc = 0
    for i in range(args.epochs):
        print("Epoch: {} ################################".format(i))
        train_loss, train_acc = trainer.train(train_loader)
        dev_loss, dev_acc = trainer.evaluate(dev_loader)
        tloss.append(float("{:.2f}".format(train_loss))),tacc.append(float("{:.2f}".format(train_acc)))
        dloss.append(float("{:.2f}".format(dev_loss))),dacc.append(float("{:.2f}".format(dev_acc)))
        print("Train Loss: {:.4f} Acc: {:.4f}".format(train_loss, train_acc))
        print("Dev   Loss: {:.4f} Acc: {:.4f}".format(dev_loss, dev_acc))
        if dev_acc > best_acc:
            best_acc = dev_acc
            trainer.save(args.save_path)
        print("###########################################")

    trainer.load(args.save_path)
    test_loss, test_acc = trainer.evaluate(test_loader)
    print("Test   Loss: {:.4f} Acc: {:.4f}".format(test_loss, test_acc))
    
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    x=np.arange(0,20)
    x_major_locator=MultipleLocator(1)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.yticks(fontsize=10)
    plt.plot(x, tacc, 'o-',label="Train_Accuracy")
    plt.plot(x, dacc, 'o-',label="Valid_Accuracy")
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.yticks(fontsize=10)
    ax=plt.gca()
    yminorLocator= MultipleLocator(0.05)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.plot(x, tloss, '.-',label="Train_Loss")
    plt.plot(x, dloss, '.-',label="Valid_Loss")
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.legend(loc='best')
    plt.savefig("./cnn/best.png")
    plt.show()
    