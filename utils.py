import numpy as np
import torch
from torch.distributions import Normal, Laplace, Gumbel
import GPy
import igraph as ig
import pandas as pd
import uuid
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from cdt.metrics import retrieve_adjacency_matrix


class Dist:
    def __init__(self, d, noise_std=1, noise_type='Gauss', adjacency=None, GP=True, lengthscale=1, f_magn=1, GraNDAG_like=False):
        self.d = d
        self.GP = GP
        self.lengthscale = lengthscale
        self.f_magn = f_magn
        self.GraNDAG_like = GraNDAG_like

        noise_std = noise_std * torch.ones(d) if isinstance(noise_std, (int, float)) else noise_std
        self.noise = self._init_noise(noise_type, noise_std)

        self.adjacency = adjacency or self._initialize_adjacency(d)

    @staticmethod
    def _initialize_adjacency(d):
        adjacency = np.ones((d, d))
        adjacency[np.tril_indices(d)] = 0
        assert np.allclose(adjacency, np.triu(adjacency))
        return adjacency

    @staticmethod
    def _init_noise(noise_type, noise_std):
        if noise_type == 'Gauss':
            return Normal(0, noise_std)
        if noise_type == 'Laplace':
            return Laplace(0, noise_std / np.sqrt(2))
        if noise_type == 'Gumbel':
            return Gumbel(0, np.sqrt(noise_std) * np.sqrt(6) / np.pi)
        raise NotImplementedError("Unknown noise type.")

    def sampleGP(self, X, lengthscale=1):
        ker = GPy.kern.RBF(input_dim=X.shape[1], lengthscale=lengthscale, variance=self.f_magn)
        return np.random.multivariate_normal(np.zeros(len(X)), ker.K(X, X))

    def sample(self, n):
        noise = self.noise.sample((n,))
        X = torch.zeros(n, self.d)
        noise_var = np.zeros(self.d)

        for i in range(self.d):
            parents = np.nonzero(self.adjacency[:, i])[0]
            if self.GraNDAG_like:
                noise_var[i] = np.random.uniform(1, 2) if len(parents) == 0 else np.random.uniform(0.4, 0.8)
                X[:, i] = np.sqrt(noise_var[i]) * noise[:, i]
            else:
                X[:, i] = noise[:, i]
            if len(parents) > 0 and self.GP:
                X[:, i] += torch.tensor(self.sampleGP(np.array(X[:, parents]), self.lengthscale))
            elif len(parents) > 0:
                for j in parents:
                    X[:, i] += torch.sin(X[:, j])
        return X, noise_var


def simulate_dag(d, s0, graph_type, triu=False):
    def _random_permutation(M):
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _acyclic_orient(B_und):
        return np.triu(B_und, k=1) if triu else np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
    elif graph_type == 'SF':
        G_und = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=False)
    elif graph_type == 'BP':
        top = int(0.2 * d)
        G_und = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
    else:
        raise ValueError("Unknown graph type")
    B_und = _graph_to_adjmat(G_und)
    B = _acyclic_orient(B_und)
    return _random_permutation(B) if not triu else B


def np_to_csv(array, save_path):
    output = os.path.join(os.path.dirname(save_path), f'tmp_{uuid.uuid4()}.csv')
    pd.DataFrame(array).to_csv(output, header=False, index=False)
    return output


def pns_(model_adj, x, num_neighbors, thresh):
    for node in range(x.shape[1]):
        x_other = np.copy(x)
        x_other[:, node] = 0
        reg = ExtraTreesRegressor(n_estimators=500).fit(x_other, x[:, node])
        mask_selected = SelectFromModel(reg, threshold=f"{thresh}*mean", prefit=True, max_features=num_neighbors).get_support(indices=False)
        model_adj[:, node] *= mask_selected.astype(np.float)
    return model_adj


def edge_errors(pred, target):
    true = retrieve_adjacency_matrix(target)
    pred = retrieve_adjacency_matrix(pred, target.nodes() if isinstance(target, nx.DiGraph) else None)
    diff = true - pred
    rev = (((diff + diff.T) == 0) & (diff != 0)).sum() / 2
    return (diff == 1).sum() - rev, (diff == -1).sum() - rev, rev


def SHD(pred, target):
    return sum(edge_errors(pred, target))
