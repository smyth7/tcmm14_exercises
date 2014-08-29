from cvxpy import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class Node(object):
    def __init__(self, cost):
        self.source = Variable()
        self.cost = cost(self.source)
        self.edge_flows = []

    def constraints(self):
        """The constraint net flow == 0."""
        net_flow = sum(self.edge_flows) + self.source
        return [net_flow == 0]

class Generator(Node):
    def __init__(self, U):
        self.U = U
        super(Generator, self).__init__(square)

    def constraints(self):
        # Net flow constraint from Node class.
        constraints = super(Generator, self).constraints()
        ### Your code here ###
        pass # Replace this with your code.

class Consumer(Node):
    def __init__(self, L):
        self.L = L
        super(Consumer, self).__init__(lambda x: 0)

    def constraints(self):
        # Net flow constraint from Node class.
        constraints = super(Consumer, self).constraints()
        ### Your code here ###
        pass # Replace this with your code.

class Edge(object):
    def __init__(self, cost):
        self.flow = Variable()
        self.cost = cost(self.flow)

    def connect(self, in_node, out_node):
        """Connects two nodes via the edge."""
        in_node.edge_flows.append(-self.flow)
        out_node.edge_flows.append(self.flow)

class CapEdge(Edge):
    def __init__(self, c):
        self.c = c
        super(CapEdge, self).__init__(square)

    def constraints(self):
        ### Your code here ###
        pass # Replace this with your code.

# Define problem data.
np.random.seed(1)
grid_dim = 7
p = grid_dim*grid_dim
k = 6
n = 2*(grid_dim-1)*grid_dim
U = np.random.uniform(20, 50, size=k)
L = -np.random.uniform(0, 5, size=p)
c = np.random.uniform(5, 10, size=n)

# Create the nodes and edges.
generators = [Generator(U[i]) for i in range(k)]
consumers = [Consumer(L[i]) for i in range(k, p)]
nodes = generators + consumers
edges = [CapEdge(c[i]) for i in range(n)]

# Create a networkx graph.
# G.nodes() is a list of node keys.
# G.edges() is a list of edge keys of the form (node key, node key).
G = nx.grid_2d_graph(grid_dim, grid_dim)
# Map node key to the index in nodes.
node_key_to_idx = {key:i for i, key in enumerate(G.nodes())}

# Connect nodes via edges.
for i, key in enumerate(G.edges()):
    idx1 = node_key_to_idx[key[0]]
    idx2 = node_key_to_idx[key[1]]
    edges[i].connect(nodes[idx1], nodes[idx2])

# Solve the problem.
cost = sum([object.cost for object in nodes + edges])
obj = Minimize(cost)
constraints = []
for object in nodes + edges:
    constraints += object.constraints()
Problem(obj, constraints).solve(verbose=True)

# Plot the result.
plt.figure(figsize=(12,10))
pos = dict(zip(G,G)) # dictionary of node names->positions
node_colors = [node.source.value for node in nodes]
nodes = nx.draw_networkx_nodes(G,pos,node_color=node_colors, with_labels=False,
                               node_size=100, node_cmap=plt.cm.Reds)
edge_colors = [abs(edge.flow).value for edge in edges]
edges = nx.draw_networkx_edges(G,pos,edge_color=edge_colors,width=4,
                               edge_cmap=plt.cm.Blues, arrows=True)
plt.colorbar(edges, label='Edge flow')
plt.colorbar(nodes, label='Node source')
plt.axis('off')
plt.show()
