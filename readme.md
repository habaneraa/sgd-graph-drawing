# Graph Drawing via SGD

$(SGD)^2$ is a framework that can optimize a graph layout with any desired set of aesthetic metrics. It's full name is Scalable Graph Drawing via Stochastic Gradient Descent. The code in this repo is forked from its original author: [tiga1231/graph-drawing, branch sgd](https://github.com/tiga1231/graph-drawing/tree/sgd). The changes include:

- Removed all experimental code or notebook, and only keeps the core, which becomes a Python package `sgd2`.
- Added a new visualization option, which can render an animation of the optimization progress and save it as a GIF or MP4 file.
- Added a new aesthetic criterion: **edge orthogonality**. Its basic idea is that we hope the edges in a graph are in a group of specific directions. For example, we want the lines in a metro map to stay as vertical or horizontal as possible.
- Allows users to set a initial layout before optimization. The layout is a set of 2d coordinates attached to graph nodes.
- code comments in Chinese.

For more information about $(SGD)^2$, see https://arxiv.org/abs/2112.01571 (TVCG 2022 paper)

## Edge Orthogonality Criterion

Suppose there is an undirected edge between node P and Q. The vector from P to Q is $v$. We hope the direction of this vector $v$ is close to some specific lines, such as horizontal line or vertical line.

To get criterion function, first normalize $v$, then compute the product of all $||v \times e_i||$, where $\times$ is vector cross product.

For example, if we have $e_1 = (0, 1), e2=(1, 0)$ and normalized vector of edge $v_e=(x_e, y_e)$, then
$$
L_{EO}(G)=\sum_e |x_e y_e| / |E|
$$
where $|E|$ is the number of edges of graph $G$. Our algorithm will try to minimize this function.

## Initial Positions and Layout

Before optimization, we will check whether the nodes have attribute `init_pos`. If so, set the position of nodes to the attribute values and scale those coordinates into a proper range. It guarantees that the bounding box of those positions is a square with sides of length $\sqrt{|N|}$. This scaling operation works well when optimizing large graphs.

One can use `networkx` API to initialize a layout.

```python
G = nx.Graph()
G.add_node('a')
G.nodes['a']['init_pos'] = (1.0, 1.0)
# or from other data structures
G.add_nodes_from([(n.id, dict(init_pos=n.pos)) for n in my_nodes])
gd = GD2(G)
```

## Requirements

pytorch, networkx

## Demo

random tree, 150 nodes, use stress + ideal edge length + edge orthogonality

<video id="video" controls="" preload="none">
      <source id="mp4" src="./demos/stress-il-eo.mp4" type="video/mp4">
</videos>
