import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing

class MPL(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPL, self).__init__(aggr='add')
        self.mlp1 = torch.nn.Linear(in_channels*2, out_channels)
        self.mlp2 = torch.nn.Linear(in_channels, out_channels)
        self.mlp3 = torch.nn.Linear(2*out_channels, 1)
        self.mlp4 = torch.nn.Linear(2*out_channels, 1)
        self.mlp5 = torch.nn.Linear(in_channels,16)
        self.mlp6 = torch.nn.Linear(out_channels,16)
        self.mlp7 = torch.nn.Linear(16,1)

    def forward(self, x, edge_index):

        msg = self.propagate(edge_index, x=x)
        x = F.relu(self.mlp2(x))
        w1 = F.sigmoid(self.mlp3(torch.cat([x,msg], dim=1)))
        w2 = F.sigmoid(self.mlp4(torch.cat([x,msg], dim=1)))
        out = w1*msg + w2*x
        
        return out

    def message(self, x_i, x_j, edge_index):
        msg = F.relu(self.mlp1(torch.cat([x_i, x_j-x_i], dim=1)))
        w1 = F.tanh(self.mlp5(x_i))
        w2 = F.tanh(self.mlp6(msg))
        w = self.mlp7(w1*w2)
        w = softmax(w, edge_index[0])
        
        return msg*w
    
class GNN(torch.nn.Module):
    def __init__(self, dataset, predict):
      super(GNN, self).__init__()
    
      assert dataset in ('prompt_new', 'displaced', 'prompt_old')
      assert predict in ('pT', '1/pT', 'pT_classes')
      
      self.predict = predict
      self.dataset = dataset
      
      if self.dataset in ['prompt_new', 'displaced']:
        self.conv1 = MPL(4,128 )
      if self.dataset == 'prompt_old':
        self.conv1 = MPL(3,128 )
      
      self.conv2 = MPL(128,32)
      self.conv3 = MPL(32,64 )
      self.conv4 = MPL(64,32 )
      self.lin1 = torch.nn.Linear(32*2, 128)
      self.lin2 = torch.nn.Linear(128, 16)
      self.lin3 = torch.nn.Linear(16, 16)
      
      if self.predict=='pT':
        self.lin4 = torch.nn.Linear(16, 1)
      if self.predict=='1/pT':
        self.lin4 = torch.nn.Linear(16, 1)
      if self.predict=='pT_classes':
        self.lin4 = torch.nn.Linear(16, 4)
      self.global_att_pool1 = gnn.GlobalAttention(torch.nn.Sequential(torch.nn.Linear(32, 1)))
      self.global_att_pool2 = gnn.GlobalAttention(torch.nn.Sequential(torch.nn.Linear(32, 1)))

    def forward(self, data):
      x, edge_index, batch = data.x, data.edge_index, data.batch
      x = F.relu(self.conv1(x, edge_index))
      x = F.relu(self.conv2(x, edge_index))
      x1 = self.global_att_pool1(x, batch)
      x = F.relu(self.conv3(x, edge_index))
      x = F.relu(self.conv4(x, edge_index))
      x2 = self.global_att_pool2(x, batch)
      x = torch.cat([x1, x2], dim=1)
      x = F.relu(self.lin1(x))
      x = F.relu(self.lin2(x))
      x = self.lin3(x)
      x = self.lin4(x)
      if self.predict=='1/pT':
        x = F.sigmoid(x)
      if self.predict=='pT_classes':
        x = F.softmax(x, dim=1)
      x = x.squeeze(1)

      return x