
import math
import time
import itertools
import pickle as pkl

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import networkx as nx


from .utils import utils
from .utils import vis as V
from .utils.CrossingDetector import CrossingDetector

from . import criteria as C
from . import quality as Q


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

if is_interactive():
    from tqdm.notebook import tqdm
    from IPython import display
else:
    from tqdm import tqdm
    display = None



class GD2:
    """使用梯度下降来优化图布局
    Ahmed et al. , Multicriteria Scalable Graph Drawing via Stochastic Gradient Descent

    输入 G(V, E)
    输出 pos: v -> (x, y)
    可逐代将布局结果可视化
    """
    def __init__(self, G, device='cpu'):
        # NetworkX graph
        assert isinstance(G, nx.Graph)
        self.G = G
        
        # 1 - 图相关数据的初始化
        self.D, self.adj_sparse, self.k2i = utils.shortest_path(G)
        
        # 邻接矩阵: torch tensor
        self.adj = torch.from_numpy((self.adj_sparse+self.adj_sparse.T).toarray()).to(device)
        # 节点距离矩阵: torch tensor
        self.D = torch.from_numpy(self.D).to(device)
        # 将 索引值映射到节点键 的 dict
        self.i2k = {i:k for k,i in self.k2i.items()}
        # W
        self.W = 1/(self.D**2+1e-6)
        # 节点度
        self.degrees = np.array([self.G.degree(self.i2k[i]) for i in range(len(self.G))])
        self.maxDegree = max(dict(self.G.degree).values())

        # 图中所有边的 节点索引对
        self.edge_indices = [(self.k2i[e0], self.k2i[e1]) for e0,e1 in self.G.edges]
        self.node_indices = range(len(self.G))
        # 所有节点对的索引. 数组形状为 (_, 2) 且保证第一个节点索引小于第二个节点索引
        self.node_index_pairs = np.c_[
            np.repeat(self.node_indices, len(self.G)),
            np.tile(self.node_indices, len(self.G))
        ]
        self.node_index_pairs = self.node_index_pairs[self.node_index_pairs[:,0]<self.node_index_pairs[:,1]]
        
        self.node_edge_pairs = list(itertools.product(self.node_indices, self.edge_indices))
        
        incident_edge_groups = [
            [(G.degree(k), self.k2i[k], self.k2i[n])
            for n in G.neighbors(k)]
            for k in G.nodes
        ]
        incident_edge_pairs = [
            [
                (i[0],)+i[1:]+j[1:] 
                for i,j in itertools.product(ieg, ieg) 
                if i<j
            ] 
            for ieg in incident_edge_groups
        ]
        # 所有相邻的边的对
        self.incident_edge_pairs = sum(incident_edge_pairs, [])
        ## filter out incident edge pairs
        self.non_incident_edge_pairs = [
            [self.k2i[e1[0]], self.k2i[e1[1]], self.k2i[e2[0]], self.k2i[e2[1]]] 
            for e1,e2 in itertools.product(G.edges, G.edges) 
            if e1<e2 and len(set(e1+e2))==4
        ]

        # 2 - 初始布局
        # 原始方案: 在方形范围内 随机初始化
        # 节点布局坐标tensor;  element-wise 乘以 \sqrt{|N|}
        self.pos = (len(self.G.nodes)**0.5) * torch.randn(len(self.G.nodes), 2, device=device)
        # 按 node attribute: init_pos 初始化位置
        # 覆盖tensor中原本的随机值
        do_scaling = False
        for i in self.node_indices:
            ret = G.nodes[self.i2k[i]].get('init_pos')
            if ret is not None:
                do_scaling = True
                self.pos[i, 0], self.pos[i, 1] = ret[0], ret[1]
        if do_scaling:
            minxy, maxxy = torch.amin(self.pos, dim=0), torch.amax(self.pos, dim=0)
            center = (minxy + maxxy) / 2
            rect = (maxxy - minxy) / 2
            divisor =  max(rect[0], rect[1]) / len(self.G.nodes)**0.5
            assert center.repeat(self.pos.shape[0], 1).shape == self.pos.shape
            self.pos = (self.pos - center.repeat(self.pos.shape[0], 1)) / divisor
        # pos = (pos - center) / divisor
        # 处理完 pos 数据后，再开启该tensor的requires_grad
        self.pos = self.pos.requires_grad_()

        init_pos_numpy = self.pos.detach().cpu().numpy().copy()
        init_pos_G = {k:init_pos_numpy[self.k2i[k]] for k in G.nodes}
        self.pos_history = [init_pos_G]

        
        # 3 - 优化过程中的变量
        self.qualities_by_time = []
        self.i = 0
        self.runtime = 0
        self.last_time_eval = 0 ## last time that evaluates the quality
        self.last_time_vis = 0
        self.iters = []
        self.loss_curve = []
        self.sample_sizes = {}
        
        # 4 - 基于 MLP 的交点判别器
        self.crossing_detector = CrossingDetector()
        self.crossing_detector_loss_fn = nn.BCELoss()
        self.crossing_pos_loss_fn = nn.BCELoss(reduction='sum')
        # self.crossing_detector_optimizer = optim.SGD(self.crossing_detector.parameters(), lr=0.1)
        self.crossing_detector_optimizer = optim.Adam(self.crossing_detector.parameters(), lr=0.01)

        self.device = device
    
    @property
    def pos_dict(self):
        pos_numpy = self.pos.detach().cpu().numpy().copy()
        return {k:pos_numpy[self.k2i[k]] for k in self.G.nodes}
        
    def grad_clamp(self, l, c, weight, optimizer, ref=1):
        optimizer.zero_grad()
        l.backward(retain_graph=True)
        grad = self.pos.grad.clone()
        grad_norm = grad.norm(dim=1)
        is_large = grad_norm > weight*ref
        grad[is_large] = grad[is_large] / grad_norm[is_large].view(-1,1) * weight*ref
        self.grads[c] = grad
        
    def optimize(self,
        criteria_weights={'stress':1.0}, 
        sample_sizes={'stress':128},
        evaluate=None,
        evaluate_interval=None,
        evaluate_interval_unit = 'iter',
        max_iter=int(1e4),
        time_limit=7200,
        grad_clamp=4,
        vis_interval=100,
        vis_interval_unit = 'iter',
        clear_output=False,
        optimizer_kwargs=None,
        scheduler_kwargs=None,
        criteria_kwargs=None,
    ):
        if criteria_kwargs is None:
            criteria_kwargs = {c:dict() for c in criteria_weights}
        self.sample_sizes = sample_sizes
        ## shortcut of object attributes
        G = self.G
        D, k2i = self.D, self.k2i
        i2k = self.i2k
        adj = self.adj
        W = self.W
        pos = self.pos
        degrees = self.degrees
        maxDegree = self.maxDegree
        device = self.device
        
        self.init_sampler(criteria_weights)
            
        ## measure runtime
        t0 = time.time()

        
        ## def optimizer
        if optimizer_kwargs.get('mode', 'SGD') == 'SGD':
            optimizer_kwargs_default = dict(
                lr=1, 
                momentum=0.7, 
                nesterov=True
            )
            Optimizer = optim.SGD
        elif optimizer_kwargs.get('mode', 'SGD') == 'Adam':
            optimizer_kwargs_default = dict(lr=0.0001)
            Optimizer = optim.Adam
        for k, v in optimizer_kwargs.items():
            if k != 'mode':
                optimizer_kwargs_default[k] = v
        optimizer = Optimizer([pos], **optimizer_kwargs_default)
        

#         patience = np.ceil(np.log2(len(G)+1))*100
#         if 'stress' in criteria_weights and sample_sizes['stress'] < 16:
#             patience += 100 * 16/sample_sizes['stress']
        patience = np.ceil(np.log2(len(G)+1)) * 300 * max(1, 16/min(sample_sizes.values()))
        patience = 20000
        scheduler_kwargs_default = dict(
            factor=0.9, 
            patience=patience, 
            min_lr=1e-5, 
            verbose=True
        )
        if scheduler_kwargs is not None:
            for k,v in scheduler_kwargs.items():
                scheduler_kwargs_default[k] = v
        scheduler_kwargs = scheduler_kwargs_default
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

        iterBar = tqdm(range(max_iter), desc='iter', colour='blue')

        ## smoothed loss curve during training
        s = 0.5**(1/100) ## smoothing factor for loss curve, e.g. s=0.5**(1/100) is setting 'half-life' to 100 iterations
        weighted_sum_of_loss, total_weight = 0, 0
        vr_target_dist, vr_target_weight = 1, 0
            
        ## start training
        for iter_index in iterBar:
            t0 = time.time()
            ## optimization
            loss = self.pos[0,0]*0 ## dummy loss
            self.grads = {}
            ref = 1
            for c, weight in criteria_weights.items():
                if callable(weight):
                    weight = weight(iter_index)
                    
                if weight == 0:
                    continue
                
                if c == 'stress':
                    sample = self.sample(c)
                    l = weight * C.stress(
                        pos, D, W, device,
                        sample=sample, reduce='mean')
                    loss += l
#                     self.grad_clamp(l, c, weight, optimizer, ref)
                elif c == 'ideal_edge_length':
                    sample = self.sample(c)
                    l = weight * C.ideal_edge_length(
                        pos, G, k2i, device, 
                        targetLengths=None, 
                        sampleSize=sample_sizes[c],
                        reduce='mean',
                    )
                    loss += l
#                     self.grad_clamp(l, c, weight, optimizer, ref)
                elif c == 'neighborhood_preservation':
                    sample = self.sample(c).tolist()
                    l = weight * C.neighborhood_preseration(
                        pos, G, adj, 
                        k2i, i2k, 
                        degrees, maxDegree,
                        sample=sample,
#                         n_roots=sample_sizes[c], 
                        depth_limit=2
                    )
                    loss += l
#                     self.grad_clamp(l, c, weight, optimizer, ref)
                elif c == 'crossings':
                    ## neural crossing detector
                    sample = self.sample(c)
                    sample = torch.stack(sample, 1)
                    edge_pair_pos = self.pos[sample].view(-1,8)
                    labels = utils.are_edge_pairs_crossed(edge_pair_pos)
                    
#                     edge_pair_pos = self.pos[sample].view(-1,8)
#                     labels = utils.are_edge_pairs_crossed(edge_pair_pos)
#                     if labels.sum() < 1:
#                         sample = torch.cat([sample, self.sample_crossings(c)], dim=0)
#                         edge_pair_pos = self.pos[sample].view(-1,8)
#                         labels = utils.are_edge_pairs_crossed(edge_pair_pos)
                        
                    ## train crossing detector
                    self.crossing_detector.train()
                    for _ in range(2):
                        preds = self.crossing_detector(edge_pair_pos.detach().to(device)).view(-1)
                        loss_nn = self.crossing_detector_loss_fn(
                            preds, 
                            (
#                                 labels.float()*0.7+0.15 + 0.10*(2*torch.rand(labels.shape)-1)
                                labels.float()
                            ).to(device)
                        )
                        self.crossing_detector_optimizer.zero_grad()
                        loss_nn.backward()
                        self.crossing_detector_optimizer.step()
                    
                    ## loss of crossing
                    self.crossing_detector.eval()
                    preds = self.crossing_detector(edge_pair_pos.to(device)).view(-1)
                    l = weight * self.crossing_pos_loss_fn(preds, (labels.float()*0).to(device))
                    loss += l
#                     self.grad_clamp(l, c, weight, optimizer, ref)
                    
                elif c == 'crossing_angle_maximization':
                    sample = self.sample(c)
                    sample = torch.stack(sample, dim=-1)
                    pos_segs = pos[sample.flatten()].view(-1,4,2)
                    sample_labels = utils.are_edge_pairs_crossed(pos_segs.view(-1,8))
                    if sample_labels.sum() < 1:
                        sample = self.sample_crossings(c)
                        pos_segs = pos[sample.flatten()].view(-1,4,2)
                        sample_labels = utils.are_edge_pairs_crossed(pos_segs.view(-1,8))
                    
                    l = weight * C.crossing_angle_maximization(
                        pos, G, k2i, i2k,
                        sample = sample,
                        sample_labels = sample_labels,
#                         sampleSize=sample_sizes[c], 
#                         sampleOn='crossings') ## SLOW for large sample size
#                         sampleOn='edges'
                    )
                    loss += l
#                     self.grad_clamp(l, c, weight, optimizer, ref)
                elif c == 'aspect_ratio':
                    sample = self.sample(c)
                    l = weight * C.aspect_ratio(
                        pos, sample=sample,
                        **criteria_kwargs.get(c)
                    )
                    loss += l
#                     self.grad_clamp(l, c, weight, optimizer, ref)
                elif c == 'angular_resolution':
#                     sample = [self.i2k[i.item()] for i in self.sample(c)]
                    sample = self.sample(c)
                    l = weight * C.angular_resolution(
                        pos, G, k2i, 
                        sampleSize=sample_sizes[c],
                        sample=sample,
                    )
                    loss += l
#                     self.grad_clamp(l, c, weight, optimizer, ref)
                elif c == 'vertex_resolution':
                    sample = self.sample(c)
                    l, vr_target_dist, vr_target_weight = C.vertex_resolution(
                        pos, 
                        sample=sample, 
                        target=1/len(G)**0.5, 
                        prev_target_dist=vr_target_dist,
                        prev_weight=vr_target_weight
                    )
                    l = weight * l
                    loss += l
#                     self.grad_clamp(l, c, weight, optimizer, ref)
                elif c == 'gabriel':
                    sample = self.sample(c)
                    l = weight * C.gabriel(
                        pos, G, k2i, 
                        sample=sample,
#                         sampleSize=sample_sizes[c],
                    )
                    loss += l
#                     self.grad_clamp(l, c, weight, optimizer, ref)
                elif c == 'edge_orthogonality':
                    sample = self.sample(c)
                    l = weight * C.edge_orthogonality(pos, G, k2i, sample=sample)
                    loss += l
                else:
                    print(f'Criteria not supported: {c}')

            optimizer.zero_grad()
            loss.backward()
            # 将梯度限制在一区间范围内 防止单次更新过多
            pos.grad.clamp_(-grad_clamp, grad_clamp)
            optimizer.step()
            
            self.runtime += time.time() - t0
            if self.runtime > time_limit:
                qualities = self.evaluate(qualities=evaluate)
                self.qualities_by_time.append(dict(
                    time=self.runtime,
                    iter=self.i,
                    qualities=qualities
                ))
                break
            
            ## do visualization
            pos_numpy = pos.detach().cpu().numpy().copy()
            pos_G = {k:pos_numpy[k2i[k]] for k in G.nodes}
            self.pos_history.append(pos_G)
            if (vis_interval is not None and vis_interval>0) and (
                ## option 1: if unit='iter'
                (vis_interval_unit == 'iter' and (self.i%vis_interval == 0 or self.i == max_iter-1)) 
                or  
                ## option 2: if unit='sec'
                (vis_interval_unit == 'sec' and (
                    self.runtime - self.last_time_vis>= vis_interval
                ))
            ):
                if display is not None and clear_output:
                    display.clear_output(wait=True)
                V.plot(
                    G, pos_G,
                    self.loss_curve, 
                    self.i, self.runtime, 
                    node_size=0,
                    edge_width=0.6,
                    show=True, save=False
                )
                self.last_time_vis = self.runtime


            if self.i % 100 == 99:
                iterBar.set_postfix({'loss': loss.item(), })    
            
            ## compute current loss as an expoexponential moving average  for lr scheduling
            ## and loss curve visualization
            weighted_sum_of_loss = weighted_sum_of_loss*s + loss.item()
            total_weight = total_weight*s+1
            if self.i % 10 == 0:
                self.loss_curve.append(weighted_sum_of_loss/total_weight)
                self.iters.append(self.i)
                if scheduler is not None:
                    scheduler.step(self.loss_curve[-1])


            lr = optimizer.param_groups[0]['lr']
            if lr <= scheduler.min_lrs[0]:
                break

                
            ## if eval in enabled, do eval
            if (evaluate_interval is not None and evaluate_interval>0) and (
                ## option 1: if unit='iter'
                (evaluate_interval_unit == 'iter' and (self.i%evaluate_interval == 0 or self.i == max_iter-1)) 
                or
                ## option 2: if unit='sec'
                (evaluate_interval_unit == 'sec' and (
                    self.runtime - self.last_time_eval >= evaluate_interval or self.i == max_iter-1
                ))
            ):
                qualities = self.evaluate(qualities=evaluate)
                self.qualities_by_time.append(dict(
                    time=self.runtime,
                    iter=self.i,
                    qualities=qualities
                ))
                self.last_time_eval = self.runtime
                
            self.i += 1
        ## end training loop.
        
        ## attach pos to G.nodes
        pos_numpy = pos.detach().numpy()
        for k in G.nodes:
            G.nodes[k]['pos'] = pos_numpy[k2i[k],:]
        
        ## prepare result
        return self.get_result_dict(evaluate, sample_sizes)
        # end optimize()
    
    def rotate_singular_vector(self, target, center=None):
        """2 dimensional only

        Params:
            target: 将第一个 singular vector 旋转到的位置
            center: 指定中心坐标作为 node vector 的起始点
        """
        pos_numpy = self.pos.detach().numpy()
        if center is None:
            center = pos_numpy.mean(axis=0)
        assert center.shape == (2,)

        u, s, vh = np.linalg.svd(pos_numpy - center)  # 其中 vh 应为二维平面上的singular vectors
        norm_prod = np.linalg.norm(vh[0]) * np.linalg.norm(target)
        sinr = np.cross(vh[0], target) / norm_prod
        cosr = np.dot(vh[0], target) / norm_prod
        rotation_matrix = np.array([
            [cosr, -sinr],
            [sinr, cosr],
        ])
        # Mpos' = Mr * Mpos^T, shape: (2,2) * (2,4)
        # new_pos = np.transpose(np.matmul(rotation_matrix, np.transpose(pos_numpy)))
        new_pos = np.matmul(pos_numpy, np.transpose(rotation_matrix))
        self.pos = torch.tensor(new_pos, device=self.device, requires_grad=True)




    def init_sampler(self, criteria_weights):
        # 对于每个criteria，都分别构建一个pytorch DataLoader对象，然后有顺序采样器从其中采样
        # 事实上这里的sampler使用的是迭代器对象
        self.samplers = {}
        self.dataloaders = {}
        for c, w in criteria_weights.items():
            if w == 0:
                continue
            if c == 'stress':
                # 使dataset内的数据类型为long, 采样后的sample将作为tensor indices
                node_index_pairs_tensor = torch.tensor(self.node_index_pairs, dtype=torch.long, device=self.device)
                self.dataloaders[c] = DataLoader(
                    # self.node_index_pairs, 
                    node_index_pairs_tensor,
                    batch_size=self.sample_sizes[c],
                    shuffle=True)
            elif c == 'ideal_edge_length':
                self.dataloaders[c] = DataLoader(
                    self.edge_indices, 
                    batch_size=self.sample_sizes[c], 
                    shuffle=True)
            elif c == 'neighborhood_preservation':
                self.dataloaders[c] = DataLoader(
                    range(len(self.G.nodes)), 
                    batch_size=self.sample_sizes[c], 
                    shuffle=True)
            elif c == 'crossings':
                self.dataloaders[c] = DataLoader(
                    self.non_incident_edge_pairs, 
                    batch_size=self.sample_sizes[c], 
                    shuffle=True)
            elif c == 'crossing_angle_maximization':
                self.dataloaders[c] = DataLoader(
                    self.non_incident_edge_pairs, 
                    batch_size=self.sample_sizes[c], 
                    shuffle=True)
            elif c == 'aspect_ratio':
                self.dataloaders[c] = DataLoader(
                    range(len(self.G.nodes)), 
                    batch_size=self.sample_sizes[c],
                    shuffle=True
                )
            elif c == 'angular_resolution':
                self.dataloaders[c] = DataLoader(
                    self.incident_edge_pairs, 
                    batch_size=self.sample_sizes[c],
                    shuffle=True)
            elif c == 'vertex_resolution':
                node_index_pairs_tensor = torch.tensor(self.node_index_pairs, dtype=torch.long, device=self.device)
                self.dataloaders[c] = DataLoader(
                    node_index_pairs_tensor, 
                    batch_size=self.sample_sizes[c],
                    shuffle=True)
            elif c == 'gabriel':
                self.dataloaders[c] = DataLoader(
                    self.node_edge_pairs, 
                    batch_size=self.sample_sizes[c],
                    shuffle=True
                )
            elif c == 'edge_orthogonality':
                self.dataloaders[c] = DataLoader(
                    self.edge_indices, 
                    batch_size=self.sample_sizes[c],
                    shuffle=True
                )
    
    def sample_crossings(self, c='crossing_angle_maximization', mode='use_existing_crossings'):
        if not hasattr(self, 'crossing_loaders'):
            self.crossing_loaders = {}    
        if not hasattr(self, 'crossing_samplers'):
            self.crossing_samplers = {}
        
        if mode == 'new' or c not in self.crossing_loaders:
            crossing_segs = utils.find_crossings(self.pos, list(self.G.edges), self.k2i)
            self.crossing_loaders[c] = DataLoader(crossing_segs, batch_size=self.sample_sizes[c])
            self.crossing_samplers[c] = iter(self.crossing_loaders[c])
#             print(f'finding new crossings...{crossing_segs.shape}')
        try:
            sample = next(self.crossing_samplers[c])
        except StopIteration:
            sample = self.sample_crossings(c, mode='new')
        return sample

    
    def sample(self, criterion):
        if criterion not in self.samplers:
            self.samplers[criterion] = iter(self.dataloaders[criterion])
        try:
            sample = next(self.samplers[criterion])
        except StopIteration:
            self.samplers[criterion] = iter(self.dataloaders[criterion])
            sample = next(self.samplers[criterion])
        return sample
    
    def get_result_dict(self, evaluate, sample_sizes):
        return dict(
            iter=self.i, 
            loss_curve=self.loss_curve,
            runtime=self.runtime,
            qualities_by_time=self.qualities_by_time,
            # qualities=self.qualities_by_time[-1]['qualities'],
            sample_sizes=self.sample_sizes,
            pos=self.pos,
        )
    
    
    def evaluate(
        self,
        pos=None,
        qualities={'stress'},
        verbose=False,
        mode='original'
    ):
        if pos is None:
            pos = self.pos
            
        qualityMeasures = dict()
        if qualities == 'all':
            qualities = {
                'stress',
                'ideal_edge_length',
                'neighborhood_preservation',
                'crossings',
                'crossing_angle_maximization',
                'aspect_ratio',
                'angular_resolution',
                'vertex_resolution',
                'gabriel',
            }

        for q in qualities:
            if verbose:
                print(f'Evaluating {q}...', end='')
                
            t0 = time.time()
            if q == 'stress':
                if mode == 'original':
                    qualityMeasures[q] = Q.stress(pos, self.D, self.W, self.device, None)
                elif mode == 'best_scale':
                    s = utils.best_scale_stress(pos, self.D, self.W)
                    qualityMeasures[q] = Q.stress(pos*s, self.D, self.W, self.device, None)
                    
            elif q == 'ideal_edge_length':
                if mode == 'original':
                    qualityMeasures[q] = Q.ideal_edge_length(pos, self.G, self.k2i, self.device)
                elif mode == 'best_scale':
                    s = utils.best_scale_ideal_edge_length(pos, self.G, self.k2i)
                    qualityMeasures[q] = Q.ideal_edge_length(s*pos, self.G, self.k2i, self.device)

            elif q == 'neighborhood_preservation':
                qualityMeasures[q] = 1 - Q.neighborhood_preservation(
                    pos, self.G, self.adj, self.i2k)
            elif q == 'crossings':
                qualityMeasures[q] = Q.crossings(pos, self.edge_indices)
            elif q == 'crossing_angle_maximization':
                qualityMeasures[q] = Q.crossing_angle_maximization(
                    pos, self.G.edges, self.k2i)
            elif q == 'aspect_ratio':
                qualityMeasures[q] = 1 - Q.aspect_ratio(pos)
            elif q == 'angular_resolution':
                qualityMeasures[q] = 1- Q.angular_resolution(pos, self.G, self.k2i)
            elif q == 'vertex_resolution':
                qualityMeasures[q] = 1 - Q.vertex_resolution(pos, target=1/len(self.G)**0.5)
            elif q == 'gabriel':
                qualityMeasures[q] = 1 - Q.gabriel(pos, self.G, self.k2i)
            
            if verbose:
                print(f'done in {time.time()-t0:.2f}s')
        return qualityMeasures
    
    def save_layout(self, filepath='graph_layout.pkl'):
        with open(filepath, 'wb') as f:
            pkl.dump(dict(nx_graph=self.G, pos=self.pos_dict,), f)

    def save_result(self, fn='result.pkl'):
        with open(fn, 'wb') as f:
            pkl.dump(dict(
                G=self.G,
                pos=self.pos,
                i2k=self.i2k,
                k2i=self.k2i,
                iter=self.i,
                runtime=self.runtime,
                loss_curve=self.loss_curve,
                qualities_by_time = self.qualities_by_time,
            ), f)
        