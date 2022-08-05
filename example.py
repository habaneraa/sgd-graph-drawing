import random
import pickle
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt

from sgd2 import utils, GD2
from sgd2.utils import weight_schedule as ws
from sgd2.graph_animation import visualize_animation, visualize_single

plt.style.use('seaborn')


seed = 567
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ----- 建图 ----- 

# load from mat file
# graph_name = 'dwt_307'
# mat_dir = 'input_graphs/SuiteSparse Matrix Collection'
# G = utils.load_mat(f'{mat_dir}/{graph_name}.mat')
# G = nx.grid_graph(dim=(10, 10))
# G = nx.balanced_tree(3, 4)
G = nx.random_tree(150)
# G = nx.octahedral_graph()
max_iter = int(150 * len(G.nodes))

#  ----- 设置 criteria 及其权重和采样大小 ----- 

criteria_weights = dict(
    stress=ws.SmoothSteps([max_iter/4, max_iter], [1, 0.05]),
    ideal_edge_length=ws.SmoothSteps([0, max_iter*0.2, max_iter*0.6, max_iter], [0, 0, 0.05, 0]),
    # aspect_ratio=ws.SmoothSteps([0, max_iter*0.2, max_iter*0.6, max_iter], [0, 0, 0.01, 0]),
    # angular_resolution=ws.SmoothSteps([0, max_iter*0.2, max_iter*0.6, max_iter], [0, 0, 0.001, 0]),
    # vertex_resolution=ws.SmoothSteps([0, max_iter*0.2, max_iter*0.6, max_iter], [0, 0, 0.01, 0]),
    edge_orthogonality=ws.SmoothSteps([0, max_iter*0.2, max_iter*0.6, max_iter], [0, 0, 0.01, 0]),
)
criteria = list(criteria_weights.keys())
# plot_weight(criteria_weights, max_iter)
# plt.close()

default_sample_sizes = dict(
    stress=32,
    ideal_edge_length=32,
    neighborhood_preservation=16,
    crossings=128,
    crossing_angle_maximization=64,
    aspect_ratio=max(128, int(len(G)**0.5)),
    angular_resolution=16,
    vertex_resolution=max(256, int(len(G)**0.5)),
    gabriel=64,
    edge_orthogonality=32,
)
sample_sizes = {c:default_sample_sizes[c] for c in criteria}

# ----- 执行 sgd 优化 ----- 

gd = GD2(G)
result = gd.optimize(
    criteria_weights=criteria_weights, 
    sample_sizes=sample_sizes,
    
    # evaluate='all',
    evaluate=set(criteria),
    
    max_iter=max_iter,
    time_limit=3600, ##sec
    
    evaluate_interval=max_iter, evaluate_interval_unit='iter',
    vis_interval=-1, vis_interval_unit='sec',
    
    clear_output=True,
    grad_clamp=20,
    criteria_kwargs = dict(aspect_ratio=dict(target=[1,1]),),
    optimizer_kwargs = dict(mode='SGD', lr=2),
    scheduler_kwargs = dict(verbose=True),
)

# ----- 输出 & 可视化 ----- 

animation = visualize_animation(G, gd.pos_history,
    target_length=5
)
plt.show()

visualize_single(G, gd.pos_dict)

# vis.plot(
#     gd.G, pos_G,
#     [gd.iters, gd.loss_curve], 
#     result['iter'], result['runtime'],
#     criteria_weights, max_iter,
#     # show=True, save=False,
#     node_size=1,
#     edge_width=1,
# )
# plt.show()