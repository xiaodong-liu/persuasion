import torch
import torch.nn as nn
import torch.nn.functional as F

class Receiver_Model(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Receiver_Model, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.linear = nn.Linear(s_dim, a_dim)
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.linear(x)
    
    def __criterion__(self, logits, labels):
        log_softmax = F.log_softmax(logits, dim = -1)
        return -torch.mean(torch.sum(log_softmax * labels, dim = -1))
    
    def train(self, dataloader_train, dataloader_val, learning_rate, epochs, early_stopper):
        opt_rm = torch.optim.RMSprop(self.parameters(), lr = learning_rate)
        criterion = self.__criterion__
        train_losses = []
        for epoch in range(epochs):
            train_loss = self.__train_one_epoch__(dataloader_train, opt_rm, criterion)
            val_loss = self.__val_one_epoch__(dataloader_val, criterion)
            train_losses.append(train_loss)
            if early_stopper.early_stop(val_loss):
                break
        return train_losses
            
    def __train_one_epoch__(self, dataloader_train, opt_rm, criterion):
        train_loss = 0
        for i, (feature, label) in enumerate(dataloader_train):
            pred_logit = self.forward(feature)
            loss = criterion(pred_logit, label)
            opt_rm.zero_grad()
            loss.backward()
            opt_rm.step()
            train_loss += loss.item()
        return (train_loss / (i+1))
    
    def __val_one_epoch__(self, dataloader_val, criterion):
        with torch.no_grad():
            val_loss = 0
            for i, (feature, label) in enumerate(dataloader_val):
                pred_logit = self.forward(feature)
                loss = criterion(pred_logit, label)
                val_loss += loss.item()
            return (val_loss / (i+1))

