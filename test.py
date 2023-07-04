import torch

class LinearLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.W = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.b = torch.nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x):
        # x: (...input_dim)
        return torch.einsum('...i,io,o->...o', x, self.W, self.b)
    
model = LinearLayer(3, 4)
x = torch.randn(2, 3)
print(model(x).shape) # (2, 4)