import torch.nn as nn
import torch.nn.functional as F
import torch

class Sender(nn.Module):
    def __init__(self, s_dim, m_dim, prior, utility):
        super(Sender, self).__init__()

        self.s_dim = s_dim
        self.m_dim = m_dim
        self.prior = torch.FloatTensor(prior)
        self.utility = torch.FloatTensor(utility)

        self.linears = nn.ModuleList([nn.Linear(1, m_dim) for i in range(self.s_dim)])

    def forward(self, x):
        out = torch.stack([linear(x) for linear in self.linears])
        return F.softmax(out, dim=-1)
    
    def get_loss(self, receiver_model):
        commitment = torch.transpose(self.forward(torch.FloatTensor([1])), 0, 1)
        with torch.no_grad():
            action_logits = receiver_model(commitment)
        actions = F.softmax(action_logits, dim=-1)
        loss = -torch.sum(commitment * self.prior * torch.mm(actions, torch.transpose(self.utility, 0, 1)))
        return loss
    
    def update_sender(self, receiver_model, iterations, early_stopper, opt_s):
        losses = []
        for i in range(iterations):
            loss = self.get_loss(receiver_model)
            opt_s.zero_grad()
            loss.backward()
            opt_s.step()
            losses.append(loss.item())
            if(early_stopper.early_stop(loss.item())):
                break
        return losses