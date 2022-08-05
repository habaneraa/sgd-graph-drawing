
"""图布局可视化"""
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import networkx as nx
import numpy as np


def visualize_animation(G, pos_list, 
    node_size=100.0, 
    edge_width=2.0,
    target_length=10.0,
    target_fps=50,
    auto_rescale=True,
    margin_distance=1.0,
    save_video=True,
    verbose=True
):
    """
    Params:
        auto_rescale: 自动伸缩坐标轴 以使图完整占满画面
    Ref:
        https://stackoverflow.com/questions/18229563/using-networkx-with-matplotlib-artistanimation
    """
    iters = len(pos_list) - 1
    step = math.ceil(len(pos_list) / (target_length * target_fps))
    last_frame_repeat = [pos_list[-1]] * target_fps
    pos_list = pos_list[::step] + last_frame_repeat
    actual_length = len(pos_list)/target_fps

    if verbose:
        print('rendering animation: \n' +
            f'    iter per sec: {iters/actual_length:.2f}\n' + 
            f'    frames: {len(pos_list)}')

    fig = plt.figure(figsize=[8,8])
    plt.plot()
    ax = plt.subplot(111)

    # nodes: PathCollection object,  edges: LineCollection object
    nodes = nx.draw_networkx_nodes(G, pos_list[0], ax=ax, node_size=node_size)
    edges = nx.draw_networkx_edges(G, pos_list[0], ax=ax, width=edge_width)
    # 标题文本 显示迭代轮次
    # title_text = ax.text(0.5, 1.100, f"Graph Layout, Iter: 0 / ?", 
    #     transform=ax.transAxes, ha="center")

    plt.axis('equal')
    fig.tight_layout()
    ax.autoscale(enable=auto_rescale, axis='both', tight=True)

    def update(frame):
        # 更新图表数据 每帧被调用一次
        pos = pos_list[frame]  # dict: node -> pos
        pos_array = np.array([pos[k] for k in G.nodes])
        nodes.set_offsets(pos_array)  # shape: N, 2
        edge_array = [(pos[e0],pos[e1]) for e0,e1 in G.edges]
        edges.set_segments(edge_array)
        # title_text.set_text(str(frame) + ' / ' + str(len(pos_list)))
        # 基于当前的点坐标范围 auto rescale
        if auto_rescale:
            min_x, min_y = min(pos_array[:,0]), min(pos_array[:,1])
            max_x, max_y = max(pos_array[:,0]), max(pos_array[:,1])
            ax.set_xlim(min_x-margin_distance, max_x+margin_distance)
            ax.set_ylim(min_y-margin_distance, max_y+margin_distance)
        return nodes, edges

    anim = animation.FuncAnimation(fig, update, 
        range(len(pos_list)), 
        interval=1.0/target_fps, 
        blit=True, 
        repeat=True, 
        repeat_delay=5000
        )
    if save_video:
        save_path = 'demos/save.mp4'
        def save_progress(current_frame: int, total_frames: int):
            print(f'\r  Saving: {current_frame} / {total_frames}', end='')
        print(f'animation video save path: {save_path}\n')
        anim.save(save_path, fps=target_fps, progress_callback=save_progress)
        print('done.')
    
    return anim


def visualize_single(G, pos, 
    node_size=100.0, 
    edge_width=2.0, ):

    fig = plt.figure(figsize=[8,8])
    plt.plot()
    ax = plt.subplot(111)

    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size)
    edges = nx.draw_networkx_edges(G, pos, ax=ax, width=edge_width)
    title_text = ax.set_title('Graph Layout', loc='center')
    # title_text = ax.text(0.5, 1.100, f"Graph Layout", 
    #     transform=ax.transAxes, ha="center")

    plt.axis('equal')
    fig.tight_layout()
    ax.autoscale(enable=True, axis='both', tight=True)

    plt.savefig('graph.png', dpi=300)
    plt.show()


