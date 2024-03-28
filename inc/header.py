import os, os.path as osp
import time
import pickle as pkl
import random
from tqdm import tqdm, trange
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

## Utilities ##

class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
