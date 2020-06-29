
import torch 
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

from utils import SEED, set_seed

set_seed(SEED)


class MLPRegressor(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLPRegressor, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_feature, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_output),
        )

    def forward(self, x):
        x = self.network(x)
        return x

def fit_nn(model, X, y, n_epochs=50, device="cuda:0"):
    # set up params
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()  

    # set up dataloaders
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_dataset = TensorDataset(torch.tensor(X_train.to_numpy(), dtype=torch.float64), torch.tensor(y_train.to_numpy(), dtype=torch.float64))
    train_loader = DataLoader(train_dataset, batch_size=16)
    test_dataset = TensorDataset(torch.tensor(X_test.to_numpy(), dtype=torch.float64), torch.tensor(y_test.to_numpy(), dtype=torch.float64))
    test_loader = DataLoader(test_dataset, batch_size=16)
    val_loss = 0
    model.to(device)
    loop = tqdm(total=len(train_loader) * n_epochs, position=0, leave=False)
    for epoch in range(n_epochs):
        for batch, (x, y) in enumerate(train_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            model.train()
            optimizer.zero_grad()   
            prediction = model(x)
            loss = loss_func(prediction, y)
            loss.backward()       
            optimizer.step() 
            loop.set_description('epoch:{}. loss:{:.4f}, val_loss:{:.3f}, rmse:{:.3f}'.format(epoch, loss.item(), val_loss, 0))
            loop.update(1)

        val_list = []
        for batch, (x, y) in enumerate(test_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            model.eval()
            prediction = model(x)
            loss = loss_func(prediction, y)
            val_list.append(loss.item())
        val_loss = np.mean(val_list)

    return model

def predict_nn(model, X, device="cuda:0"):
    # such small batch sizes, we can do it all
    model.eval()
    train_dataset = torch.tensor(X.to_numpy(), dtype=torch.float64).to(device).float()
    # convert it back to np for analysis
    return model(train_dataset).detach().cpu().numpy()
    
    
    