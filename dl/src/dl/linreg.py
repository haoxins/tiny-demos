import plotly.express as px
import torch

import random


def synthetic_data(w, b, num):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, feats, labels):
    num = len(feats)
    indices = list(range(num))
    random.shuffle(indices)
    for i in range(0, num, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num)])
        yield feats[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, learn_rate, batch_size):
    with torch.no_grad():
        for param in params:
            param -= learn_rate * param.grad / batch_size
            param.grad.zero_()


def py_native():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    feats, labels = synthetic_data(true_w, true_b, 1000)

    # px.scatter(x=feats[:, 1], y=labels[:, 0]).show()

    batch_size = 10
    # for X, y in data_iter(batch_size, feats, labels):
    #     print(X, "\n", y)
    #     break

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    lr = 0.03
    num_epochs = 6
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, feats, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(feats, w, b), labels)
            print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")


def load_array(data_arrays, batch_size, shuffle=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle)


def py_torch():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    feats, labels = synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    data_iter = load_array((feats, labels), batch_size)

    # print(next(iter(data_iter)))
    from torch import nn

    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()

    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 6
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(feats), labels)
        print(f"epoch {epoch + 1}, loss {l:f}")


py_native()
py_torch()
