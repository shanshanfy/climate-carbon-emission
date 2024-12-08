import os
import numpy as np
import time

from baselines.score import SCORE

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

logfile = f'./Result/log.txt'
os.makedirs(os.path.dirname(logfile), exist_ok=True)

start_time = time.time()
method = SCORE()
method.config['exps'] = 0
Order_pred, Graph_pred = method.run(None, './Data/data.npy')

