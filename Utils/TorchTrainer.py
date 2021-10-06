import torch, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Dataset, Data, DataLoader

#### Pytorch Geometric Dataset ###
class TriggerDataset(Dataset):
    def __init__(self, root='./', features=None, labels=None, indexes=None, transform=None, pre_transform=None):
        super(TriggerDataset, self).__init__(root, transform, pre_transform)
        assert len(features)==len(labels)
        self.features = features
        self.labels = labels
        self.indexes = list(range(len(self.labels)))
        self.length = len(self.indexes)
        self.edge_index = torch.tensor([(0,1),(1,2),(2,3),(3,2),(2,1),(1,0)], dtype=torch.long).T

    @property
    def raw_file_names(self):
        return ['vgc']

    @property
    def processed_file_names(self):
        return ['vghv']

    def download(self):
        return None

    def process(self):
        return None

    def len(self):
        return self.length

    def get(self, idx):
        graph = self.features[idx].reshape(-1,4).T
        y = self.labels[idx]
        data = Data(x=torch.tensor(graph, dtype=torch.float), y=torch.tensor(y, dtype=torch.float), edge_index=self.edge_index)
        return data

    
####### Training Function ######

def train_gnn(model, predict, X_train, Y_train, X_test, Y_test, fold=0, epochs=50, batch_size=512, results_path='./', progress_bar=False):
    
    assert predict in ('pT', '1/pT', 'pT_classes')
    
    
    def pTLossTorch(outputs, labels):
        weights = torch.tensor(labels<80, dtype=torch.float).to(device)*labels + torch.tensor(labels>=80, dtype=torch.float).to(device)*torch.tensor(labels<160, dtype=torch.float).to(device)*labels*2.4 + torch.tensor(labels>=160, dtype=torch.float).to(device)*10
        error = weights*(((outputs-labels)/labels)**2)
        return torch.mean(error)

    def FocalLossTorch(outputs, labels):
        alpha = 0.25
        gamma = 2
        ce_loss = torch.nn.functional.cross_entropy(outputs, labels.to(torch.long), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean()
        return focal_loss
    test_index = list(X_test.index)
    X_val = X_train.reset_index(drop=True).iloc[:int(len(X_train)*0.1)].to_numpy()
    Y_val = Y_train.reset_index(drop=True).iloc[:int(len(Y_train)*0.1)].to_numpy()
    X_train = X_train.reset_index(drop=True).iloc[int(len(X_train)*0.1):].reset_index(drop=True).to_numpy()
    Y_train = Y_train.reset_index(drop=True).iloc[int(len(Y_train)*0.1):].reset_index(drop=True).to_numpy()
    X_test = X_test.reset_index(drop=True).to_numpy()
    Y_test = Y_test.reset_index(drop=True).to_numpy()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_loader = DataLoader(TriggerDataset('./',X_train, Y_train), batch_size=batch_size, shuffle=True, num_workers = 4) 
    val_loader = DataLoader(TriggerDataset('./',X_val, Y_val), batch_size=batch_size) 
    test_loader = DataLoader(TriggerDataset('./',X_test, Y_test), batch_size=batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=1, factor=0.5)
    
    m_train_loss = []
    m_val_loss = []
    m_test_loss = []
    min_val_loss = float('inf')
    mse = torch.nn.MSELoss()
    bce = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
      train_loss = 0
      val_loss = 0
      if progress_bar:
          pbar = tqdm(train_loader)
      else:
          pbar = train_loader
      for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        labels = data.y
        if predict=='pT':
            loss = pTLossTorch(outputs, labels)
        if predict=='1/pT':
            loss = mse(outputs, labels)
        if predict=='pT_classes':
            loss = bce(outputs, labels.type(torch.long))
        loss.backward()
        optimizer.step()
        if progress_bar:
          pbar.set_description('Loss: '+str(loss.cpu().detach().numpy()))
        train_loss += loss.cpu().detach()/len(train_loader)

      for data in val_loader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        labels = data.y
        if predict=='pT':
            loss = pTLossTorch(outputs, labels)
        if predict=='1/pT':
            loss = mse(outputs, labels)
        if predict=='pT_classes':
            loss = bce(outputs, labels.type(torch.long))
        val_loss += loss.cpu().detach()/len(val_loader)
      if val_loss.detach().numpy()<min_val_loss:
        min_val_loss = val_loss.cpu().detach().numpy()
        torch.save(model.state_dict(), 'model.pth')
      lr_scheduler.step(val_loss)
      print('Epoch: ', str(epoch+1)+'/'+str(epochs),'| Training Loss: ', train_loss.numpy(), '| Validation Loss: ', val_loss.numpy())
      m_train_loss.append(train_loss.numpy())
      m_val_loss.append(val_loss.numpy())
      if epoch>20 and min(m_val_loss[-7:])>min_val_loss+0.0001:
        break

    model.load_state_dict(torch.load('model.pth'))
    test_loss = 0
    true = []
    preds = []
    
    for data in test_loader:
      data = data.to(device)
      optimizer.zero_grad()
      outputs = model(data)
      labels = data.y
      true += list(labels.detach().numpy())
      preds += list(outputs.detach().numpy())
      if predict=='pT':
        loss = pTLossTorch(outputs, labels)
      if predict=='1/pT':
        loss = mse(outputs, labels)
      if predict=='pT_classes':
        loss = bce(outputs, labels.type(torch.long))
      test_loss += loss/len(test_loader)
    
    print('Test Loss: ', test_loss.detach().numpy())
    
    OOF_preds = pd.DataFrame()
    OOF_preds['true_value'] = true
    if predict in ('pT', '1/pT'):
        OOF_preds['preds'] = preds
    else:
        preds = np.array(preds)
        OOF_preds['0-10'] = preds[:,0].reshape(-1)
        OOF_preds['10-30'] = preds[:,1].reshape(-1)
        OOF_preds['30-100'] = preds[:,2].reshape(-1)
        OOF_preds['100-inf'] = preds[:,3].reshape(-1)
    OOF_preds['row'] = test_index
    OOF_preds.to_csv(os.path.join(results_path, 'OOF_preds_'+str(fold)+'.csv'), index=False)
    
    return m_train_loss, m_val_loss
    