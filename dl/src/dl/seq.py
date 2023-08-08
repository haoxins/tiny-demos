import torch
import torch.nn as nn
import plotly.express as px


def load_array(data_arrays, batch_size, shuffle=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle)


T = 1_000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
fig = px.scatter(x=time, y=x)
# fig.show()

tau = 4
feats = torch.zeros((T - tau, tau))
for i in range(tau):
    feats[:, i] = x[i : T - tau + i]
labels = x[tau:].reshape(-1, 1)

batch_size = 16
n_train = 600

train_iter = load_array((feats[:n_train], labels[:n_train]), batch_size, shuffle=True)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net


loss = nn.MSELoss(reduction="none")


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f"epoch {epoch + 1}, loss {float(l.mean()):f}")


net = get_net()
train(net, train_iter, loss, 5, 0.01)

onestep_preds = net(feats)
fig = px.scatter(x=time[tau:], y=onestep_preds.reshape((-1,)).detach().numpy())
fig.show()
