import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class DummyNN(nn.Module):
    def __init__(self, input_size: int, num_classes: int, num_layers: int, layer_size: int, activation: str):
        super(DummyNN, self).__init__()
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, layer_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            input_size = layer_size
        
        layers.append(nn.Linear(layer_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

def mlp_block(in_features, hidden_size, out_features, activation, layer_number):
    layers = []

    if layer_number == 1:
        layers.append(nn.Linear(in_features, out_features))
        return nn.Sequential(*layers)

    for _ in range(layer_number - 1):
        layers.append(nn.Linear(in_features, hidden_size))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        in_features = hidden_size
    layers.append(nn.Linear(hidden_size, out_features))
    return nn.Sequential(*layers)


# Define the MLP Search Space
class MixedOp(nn.Module):
    def __init__(self, in_features, out_features):
        super(MixedOp, self).__init__()
        hidden_sizes = [32, 64, 128, 256]
        activations = ['relu', 'tanh', 'sigmoid']
        layer_numbers = [2, 3, 4]
        self.ops = nn.ModuleList([
            mlp_block(in_features, hidden_size, out_features, activation, layer_number)
            for hidden_size in hidden_sizes
            for activation in activations
            for layer_number in layer_numbers
        ])
        self.ops.append(nn.Linear(in_features, out_features))
    
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))


class DARTS_Network(nn.Module):
    def __init__(self, in_features, num_classes):
        super(DARTS_Network, self).__init__()
        self.layer1 = MixedOp(in_features, num_classes)
        self.alphas = [
            Variable(torch.zeros(len(self.layer1.ops)), requires_grad=True),
        ]
        print("Initial alphas:", self.alphas)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.layer1(x, F.softmax(self.alphas[0], dim=-1))
        return x


class DARTS_Best(nn.Module):
    def __init__(self, model, best_arch):
        super(DARTS_Best, self).__init__()
        self.layers = nn.ModuleList([
            model.layer1.ops[best_arch[0]],
        ])

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.layers[0](x)
        return x