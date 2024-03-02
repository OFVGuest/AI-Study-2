import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class DeepNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.ll2 = nn.Linear(hidden_size, num_outputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = F.relu(self.linear(X))
        X = self.dropout(X)
        return self.ll2(X)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.losst = 0

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        pred  = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.losst = loss.item()

        self.optimizer.step()