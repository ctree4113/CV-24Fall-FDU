import matplotlib

matplotlib.use('Agg')
import time
import torch
import torch.nn as nn
import os
import numpy as np
import skimage.measure
import random
from tqdm import tqdm as tqdm
import sys
from metrics.betti_compute import betti_number


def getBetti(binaryPredict, masks, i=0, topo_size=65):
    predict_betti_number_ls = []
    groundtruth_betti_number_ls =[]
    betti_error_ls = []
    gt_dmap = masks.cuda()

    for y in range(0, gt_dmap.shape[0], topo_size):
        for x in range(0, gt_dmap.shape[1], topo_size):
            binary = binaryPredict[y:min(y + topo_size, gt_dmap.shape[0]),
                         x:min(x + topo_size, gt_dmap.shape[1])]           
            groundtruth = gt_dmap[y:min(y + topo_size, gt_dmap.shape[0]),
                          x:min(x + topo_size, gt_dmap.shape[1])]

            predict_betti_number = betti_number(binary, i=i)
            groundtruth_betti_number = betti_number(groundtruth, i=i)
            # print(predict_betti_number, groundtruth_betti_number)
            predict_betti_number_ls.append(predict_betti_number)
            groundtruth_betti_number_ls.append(groundtruth_betti_number)
            betti_error_ls.append(abs(predict_betti_number-groundtruth_betti_number))

    return np.mean(betti_error_ls)

