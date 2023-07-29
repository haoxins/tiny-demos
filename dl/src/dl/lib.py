import plotly.express as px
import numpy as np

import math


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu) ** 2)


def draw():
    x = np.arange(-7, 7, 0.01)
    params = [(0, 1), (0, 2), (3, 1)]
    fig = px.bar(x=x, y=[normal(x, mu, sigma) for mu, sigma in params])
    fig.show()


draw()
