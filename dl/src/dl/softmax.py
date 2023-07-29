import torch
import torchvision
from torch.utils import data
from torchvision import transforms

import os

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


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


# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# imgs = X.reshape(18, 28, 28)
# labels = get_fashion_mnist_labels(y)


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


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(
            f"epoch {epoch + 1}, loss {train_metrics[0]:.4f}, train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}"
        )
    train_loss, train_acc = train_metrics


def sgd(params, learn_rate, batch_size):
    with torch.no_grad():
        for param in params:
            param -= learn_rate * param.grad / batch_size
            param.grad.zero_()


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
    for X, y in train_iter:
        # print(X.shape, X.dtype, y.shape, y.dtype)
        break

    num_inputs = 784
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # X = torch.normal(0, 1, size=(2, 5))
    # X_prob = softmax(X)
    # print(X_prob, X_prob.sum(1))

    def net(X):
        # print("shapes", X.shape, W.shape, b.shape)
        return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

    def cross_entropy(y_hat, y):
        return -torch.log(y_hat[range(len(y_hat)), y])

    # y = torch.tensor([0, 2])
    # y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    # print(y_hat[[0, 1], y])
    # print(cross_entropy(y_hat, y))
    # print(accuracy(y_hat, y) / len(y))

    # print(evaluate_accuracy(net, test_iter))

    lr = 0.1

    def updater(batch_size):
        return sgd([W, b], lr, batch_size)

    num_epochs = 10
    train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    predict(net, test_iter)
