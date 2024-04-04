from raphtory import Graph
from raphtory import algorithms as algo
import pandas as pd

graph = Graph()

graph.add_node(timestamp=1, id="Alice")
graph.add_node(timestamp=1, id="Bob")
graph.add_node(timestamp=1, id="Charlie")
graph.add_edge(timestamp=2, src="Bob", dst="Charlie", properties={"weight": 5.0})
graph.add_edge(timestamp=3, src="Alice", dst="Bob", properties={"weight": 10.0})
graph.add_edge(timestamp=3, src="Bob", dst="Charlie", properties={"weight": -15.0})

print(graph)

results = [["earliest_time", "name", "out_degree", "in_degree"]]

for graph_view in graph.rolling(window=1):
    for v in graph_view.nodes:
        results.append(
            [graph_view.earliest_time, v.name, v.out_degree(), v.in_degree()]
        )

print(pd.DataFrame(results[1:], columns=results[0]))

cb_edge = graph.edge("Bob", "Charlie")
weight_history = cb_edge.properties.temporal.get("weight").items()
print(
    "The edge between Bob and Charlie has the following weight history:", weight_history
)

weight_change = cb_edge.at(2)["weight"] - cb_edge.at(3)["weight"]
print(
    "The weight of the edge between Bob and Charlie has changed by",
    weight_change,
    "pts",
)

top_node = algo.pagerank(graph).top_k(1)
print(
    "The most important node in the graph is",
    top_node[0][0],
    "with a score of",
    top_node[0][1],
)
