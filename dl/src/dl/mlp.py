import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision

x = torch.arange(-8.0, 8.0, 1, requires_grad=True)
y = torch.relu(x)  # sigmoid(x)
y.backward(torch.ones_like(x), retain_graph=True)
# print(x.detach(), y, x.grad)

import os

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, transform=trans, download=True
    )
    print(len(mnist_train), len(mnist_test), mnist_train[0][0].shape)
    return (
        data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
        data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4),
    )


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return H @ W2 + b2


loss = nn.CrossEntropyLoss(reduction="none")


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(
            f"epoch {epoch + 1}, loss {train_metrics[0]:.4f}, train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}"
        )
    train_loss, train_acc = train_metrics


def get_fashion_mnist_labels(labels):
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [text_labels[int(i)] for i in labels]


def predict(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(1))
    titles = [true + "---" + pred for true, pred in zip(trues, preds)]
    print(titles)


if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]

    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    train(net, train_iter, test_iter, loss, num_epochs, updater)

    predict(net, test_iter)
