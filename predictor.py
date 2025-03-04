import torch
import torch.nn.functional as F
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, layers_sizes, input_size, lr, device):
        super(Predictor, self).__init__()
        self.device = device
        print(f"Using device: {torch.cuda.get_device_name(0)}({self.device}) for predictor")

        self.ls = layers_sizes
        self.in_size = input_size
        self.lr = lr
        
        # Create nn.Parameter objects instead of plain tensors
        self.Layers = nn.ParameterList()
        for layer in range(len(self.ls)):
            if layer == 0:
                self.Layers.append(nn.Parameter(
                    torch.randn(self.in_size+1, self.ls[layer], device=self.device) * 0.1
                ))
            else:
                self.Layers.append(nn.Parameter(
                    torch.randn(self.ls[layer-1]+1, self.ls[layer], device=self.device) * 0.1
                ))

    def forward(self, X):  # Fixed method name from 'foward' to 'forward'
        Xz = torch.cat((X, torch.ones((X.shape[0], 1), device=self.device)), dim=1)
        Resp = Xz
        for layer in self.Layers:
            R = torch.matmul(Resp, layer)
            Resp = torch.cat((R, torch.ones((R.shape[0], 1), device=self.device)), dim=1)
        Resp = Resp[:, :-1]

        Predictions = F.softmax(Resp, dim=1)
        return Predictions
    
    # Instead of custom zero_grad and backward methods, use an optimizer:
    def get_optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)