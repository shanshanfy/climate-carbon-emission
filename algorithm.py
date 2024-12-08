import os
import torch
import numpy as np
import pandas as pd
import uuid
import random
from cdt.utils.R import launch_R_script
from cdt.metrics import SID, SHD

import time

def log(logfile, str_):
    """ Log a string in a file """
    with open(logfile,'a') as f:
        f.write(str_+'\n')
    print(str_)

def backRE(tar_DAG, P_KCI):
    sid_val = SID(tar_DAG, P_KCI)
    shd_val = SHD(tar_DAG, P_KCI)
    precision, recall, f1 = f1_score(tar_DAG, P_KCI)
    distance = l2_distance(tar_DAG, P_KCI)
    return [sid_val, shd_val, precision, recall, f1, distance]

def f1_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    precision = true_positive / (np.sum(y_pred == 1) + 1e-10)
    recall = true_positive / (np.sum(y_true == 1) + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1

def l2_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.sqrt(np.sum((point1 - point2)**2))
    return distance

def set_seed(seed=2023):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Stein_hess(X, eta_G, eta_H, s = None):
    """
    Estimates the diagonal of the Hessian of log p_X at the provided samples points
    X, using first and second-order Stein identities
    """
    n, d = X.shape
    
    X_diff = X.unsqueeze(1)-X
    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        s = D.flatten().median()
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    
    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n)), nablaK)
    
    nabla2K = torch.einsum('kij,ik->kj', -1/s**2 + X_diff**2/s**4, K)
    return -G**2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n)), nabla2K)

def compute_top_order(X, eta_G, eta_H, normalize_var=True, dispersion="var"):
    n, d = X.shape
    order = []
    active_nodes = list(range(d))
    for i in range(d-1):
        H = Stein_hess(X, eta_G, eta_H)
        if normalize_var:
            H = H / H.mean(axis=0)
        if dispersion == "var": # The one mentioned in the paper
            l = int(H.var(axis=0).argmin())
        elif dispersion == "median":
            med = H.median(axis = 0)[0]
            l = int((H - med).abs().mean(axis=0).argmin())
        else:
            raise Exception("Unknown dispersion criterion")
        order.append(active_nodes[l])
        active_nodes.pop(l)
        X = torch.hstack([X[:,0:l], X[:,l+1:]])
    order.append(active_nodes[0])
    order.reverse()
    return order

def np_to_csv(array, save_path):
    """
    Convert np array to .csv
    array: numpy array
        the numpy array to convert to csv
    save_path: str
        where to temporarily save the csv
    Return the path to the csv file
    """
    id = str(uuid.uuid4())
    output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')

    df = pd.DataFrame(array)
    df.to_csv(output, header=False, index=False)

    return output

def cam_pruning(A, X, cutoff, prune_only=True, pns=False):
    save_path = "./Result/tmp/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data_np = np.array(X.detach().cpu().numpy())
    data_csv_path = np_to_csv(data_np, save_path)
    dag_csv_path = np_to_csv(A, save_path)

    arguments = dict()
    arguments['{PATH_DATA}'] = data_csv_path
    arguments['{PATH_DAG}'] = dag_csv_path
    arguments['{PATH_RESULTS}'] = os.path.join(save_path, "results.csv")
    arguments['{ADJFULL_RESULTS}'] = os.path.join(save_path, "adjfull.csv")
    arguments['{CUTOFF}'] = str(cutoff)
    arguments['{VERBOSE}'] = "TRUE"

    if prune_only:
        def retrieve_result():
            A = pd.read_csv(arguments['{PATH_RESULTS}']).values
            os.remove(arguments['{PATH_RESULTS}'])
            os.remove(arguments['{PATH_DATA}'])
            os.remove(arguments['{PATH_DAG}'])
            return A
        dag = launch_R_script("baselines/score/cam_pruning.R", arguments, output_function=retrieve_result)
        return dag
    else:
        def retrieve_result():
            A = pd.read_csv(arguments['{PATH_RESULTS}']).values
            Afull = pd.read_csv(arguments['{ADJFULL_RESULTS}']).values
            
            return A, Afull
        dag, dagFull = launch_R_script("baselines/score/CAM.R", arguments, output_function=retrieve_result)
        top_order = fullAdj2Order(dagFull)
        return dag, top_order

def fullAdj2Order(A):
    order = list(A.sum(axis=1).argsort())
    order.reverse()
    return order

def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d,d))
    for i, var in enumerate(top_order):
        A[var, top_order[i+1:]] = 1
    return A

def SCORE_CAM(X, eta_G=0.001, eta_H=0.001, cutoff=0.001, normalize_var=False, dispersion="var"):
    top_order = compute_top_order(X, eta_G, eta_H, normalize_var, dispersion)
    G_pred = full_DAG(top_order)
    return G_pred, cam_pruning(G_pred, X, cutoff), top_order


class SCORE(object):
    def __init__(self) -> None:
        self.config = {
                    'name': 'SCORE',
                    'eta_G': 0.001,
                    'eta_H': 0.001,
                    'cam_cutoff': 0.001, 
                    'detail': True,
                    'seed': 2023,
                    }

    def set_Configuration(self, config):
        self.config = config

    def run(self, CInd, dataPath, config=None):
        if config is None:
            config = self.config
        
        self.name = config['name']
        self.eta_G = config['eta_G']
        self.eta_H = config['eta_H']
        self.cam_cutoff = config['cam_cutoff']
        self.detail = config['detail']
        self.seed = config['seed']

        set_seed(seed=self.seed)

        start_time = time.time()
        savePath = './Result/'.format(self.name)
        os.makedirs(os.path.dirname(savePath), exist_ok=True)


        X = np.load(dataPath)
        X_torch = torch.from_numpy(X).float()
        Order_pred, Graph_pred, P_order_SCORE =  SCORE_CAM(X_torch, self.eta_G, self.eta_H, self.cam_cutoff)

        end_time = time.time()
        run_time = end_time - start_time
        print("End {}: {}s . ".format(self.name, run_time))


        file=open('./Result/data.txt','w')
        for layer in P_order_SCORE:
            file.write(str(layer))
        file.close()

        np.save("./Result/Order_pred.npy", Order_pred)
        np.save("./Result/Graph_pred.npy", Graph_pred)

    
        np.savetxt("./Result/Graph_pred.txt", Graph_pred, delimiter=',', fmt='%d') 
        np.savetxt("./Result/Order_pred.txt", Order_pred, delimiter=',', fmt='%d') 
        

        return Order_pred, Graph_pred
