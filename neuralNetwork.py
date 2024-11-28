import torch
import torch.nn.functional as F

g = torch.Generator().manual_seed(123)

class Linear:
    def __init__(self, input_size, output_size):
        self.weight = torch.randn((input_size, output_size), generator=g) / input_size
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        self.out = x @ self.weight + self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return [self.out]

class ReLu:
    def __call__(self, x):
        self.input = x
        self.out = torch.relu(x)
        return self.out

    def parameters(self):
        return [self.input, self.out]

    def backward(self, da):
        return (self.input > 0) * da

class MLP:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        yhat = x
        for layer in self.layers:
            yhat = layer(yhat)
        return yhat

    def parameters(self):
        return [p for layers in self.layers for p in layers.parameters()]

def bce_loss_with_logits_naive(inputs, targets):
    losses = -1 * (targets * inputs.log() + (1-targets) * (1-inputs).log())
    return losses.mean()

def logit_bce_loss(inputs, targets):
    """
    deal with boundary cases such as σ = 0 or 1, log(0) gracefully
    1/n * sum(y .* log.(σ) + (1 .- y).* log.(1 .- σ))
    rather you should use xlogy and xlog1py
    """
    sigmas = F.sigmoid(inputs) # apply transform on the logits
    losses = -1 * (torch.special.xlogy(targets, sigmas) + torch.special.xlog1py(1 - targets, - sigmas))
    return losses.mean()

if __name__ == '__main__':
    layer =  Linear(2, 10)
    print(f"layer weight = {layer.weight}")
    print(f"layer bias = {layer.bias}")

    x = torch.randn(2)
    z = layer(x)
    print(x, z)
