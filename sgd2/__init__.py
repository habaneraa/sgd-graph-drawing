
from .gd2 import GD2
from .utils import utils
from .utils.weight_schedule import SmoothSteps
from .graph_animation import visualize_animation, visualize_single

def default_sample_sizes(G):
    return dict(
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

import matplotlib.pyplot as plt
plt.style.use('seaborn')
