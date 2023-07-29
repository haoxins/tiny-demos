import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, "d.t")
x = torch.load("d.t")
print(x)
y = torch.zeros(4)
torch.save([x, y], "d.t")
x, y = torch.load("d.t")
print(x, y)

d = {"x": x, "y": y}
torch.save(d, "d.t")
d = torch.load("d.t")
print(d)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), "mlp.t")

clone = MLP()
clone.load_state_dict(torch.load("mlp.t"))
clone.eval()

Y_clone = clone(X)
print(Y_clone == Y)
