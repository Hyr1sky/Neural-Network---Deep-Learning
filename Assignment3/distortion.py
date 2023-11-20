import torch


class MSE(torch.nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.squared_difference = torch.nn.MSELoss(reduction='none')

    def forward(self, X, Y):
        return torch.mean(self.squared_difference(X, Y))
