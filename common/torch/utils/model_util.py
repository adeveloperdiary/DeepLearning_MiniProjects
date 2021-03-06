import torch


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


def weights_init_xavier_uniform(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def weights_init_xavier_normal(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class CNNBaseModel(torch.nn.Module):
    def __init__(self):
        super(CNNBaseModel, self).__init__()

    def weights_init_xavier_normal(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def print_network(self, X=None):
        """
            Use this function to print the network sizes.
        """
        if X is None:
            X = torch.rand(1, 3, 224, 224)
        for layer in self.model:
            X = layer(X)
            print(layer.__class__.__name__, 'Output shape:\t', X.shape)
