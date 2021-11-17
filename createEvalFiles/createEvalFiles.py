import argparse
import sys
import os
import time
import math

import random
import numpy as np

import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import (InferenceRunner, MaxSaver,
                                 Saver)
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from torchsparse.utils import sparse_collate_fn, sparse_quantize, sparse_collate_tensors
from torchsparse import SparseTensor

from core2 import builder
from core2.trainers import SemanticKITTITrainer
from core2.callbacks import MeanIoU

from os import listdir, mkdir
from os.path import isfile, join

import pandas as pd

from consts import *

def countIou(npy, index):
    tp = npy[index, index]
    fp = np.sum(npy[index, :]) - tp
    tf = np.sum(npy[:, index]) - tp
    if (fp + tp + tf ) > 0:
        return (tp / (tp + tf + fp))
    else:
        return -1.

def countAcc(npy, index):
    tp = npy[index, index]
    tf = np.sum(npy[:, index]) - tp
    if (tp + tf) > 0:
        return tp/(tp + tf)
    else:
        return -1.


if __name__ == "__main__":
    startTime = time.time()

    if not os.path.isdir(path):
        print("Creating main directory")
        os.mkdir(model_name)

    if not os.path.isdir(model_name):
        print("Creating destination path")
        os.mkdir(model_name)

    # read the model
    torch.cuda.set_device(cuda)
    configs.load(PATH_CONFIG, recursive=True)
    model = builder.make_model().to(cuda)
    
    # TODO check CUDA
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device(cuda))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # get lists of all files
    lidars = sorted([f for f in listdir(pathLidars) if isfile(join(pathLidars, f))])
    labels = sorted([f for f in listdir(pathLabels) if isfile(join(pathLabels, f))])

    # create structs for save iou
    # for general iou
    generalIouArray = np.zeros(len(lidars))
    generalAccArray = np.zeros(len(lidars))

    # for xy dist
    xyIouArray = np.zeros((len(lidars), 11, 19))
    xyAccArray = np.zeros((len(lidars), 11, 19))

    # for z dist
    zIouArray = np.zeros((len(lidars), 16, 19))
    zAccArray = np.zeros((len(lidars), 16, 19))

    numberOfScans = len(lidars)
    general_iou = np.zeros((numberOfScans, 19))
    partial_ious = np.zeros((numberOfScans, 3, 11, 19))

    for skan in range(len(lidars)):
        if skan % 100 == 0:
            print(str(skan) + " out of " + str(len(lidars)) + " files were processed in " + "{:.2f}".format((time.time() - startTime)/60) + " minutes")
        # read data
        lidar = np.fromfile(pathLidars + lidars[skan], dtype=np.float32)
        label = np.fromfile(pathLabels + labels[skan], dtype=np.int32)
        # process data
        lidar = lidar.reshape(-1, 4)
        label = LABEL_MAP[label & 0xFFFF]
        lidar = lidar[label != 19]
        label = label[label != 19]

        # TODO modify pointcloud
        block = lidar.copy()

        # spherical
        #r_grid = 0.05
        #poziom = 0.16 * np.pi / 180
        #pion = 1.2 * np.pi / 180
        #voxelSize = np.array([r_grid, poziom, pion])
        #r = np.sqrt(block[:, 0]**2 + block[:,1]**2 + block[:,2]**2)
        #phi = np.arctan2(block[:, 1], block[:, 0])  # poziom
        #theta = np.arcsin(block[:, 2] / r[:])  # pion
        #block[:, :3] = np.stack((r, phi, theta), axis = 1)

        # convert to cylinder
        #rho = np.sqrt(lidar[:, 0] ** 2 + lidar[:, 1] ** 2)
        #phi = np.arctan2(lidar[:, 1], lidar[:, 0])
        #block = np.stack((rho, phi, lidar[:, 2], lidar[:, 3]), axis=1)

        # convert asmp
        #block = lidar.copy()
        block[:, 0] = [asp.smoothTransition(i, tranStart, tranDur, tranSteps, endA, newAs, newBs, breakX) for i in block[:, 0]]
        block[:, 1] = [asp.smoothTransition(i, tranStart, tranDur, tranSteps, endA, newAs, newBs, breakX) for i in block[:, 1]]

        # convert to jump
        # block[np.absolute(block[:, 0]) > mod] = np.sign(block[np.absolute(block[:, 0]) > mod]) * (mod + (np.absolute(block[np.absolute(block[:, 0]) > mod]) -mod) / divider)
        # block[np.absolute(block[:, 1]) > mod] = np.sign(block[np.absolute(block[:, 1]) > mod]) * (mod + (np.absolute(block[np.absolute(block[:, 1]) > mod]) -mod) / divider)

        # TODO modify voxelize process
        # get rounded coordinates

        # no changes
        coords = np.round(block[:, :3] / voxelSize)
        coords -= coords.min(0, keepdims=1)

        # diffrent grid size 02 05 20
        #lim1 = 10
        #lim2 = 40
        #gridSize1 = 0.2
        #gridSize2 = 0.05
        #gridSize3 = 0.02

        #coords = block.copy()

        #coords[coords >= lim2] = 900 + np.round((coords[coords>=lim2]) / gridSize3)
        #coords[np.logical_and(coords >= lim1, coords < lim2)] = 300 +  np.round((coords[np.logical_and(coords >= lim1, coords < lim2)]) / gridSize2)
        #coords[coords < lim1] = np.round(coords[coords<lim1] / gridSize1)

        # diff gitd 05 10
        #coords = block.copy()
        #lim1 = 10
        #lim2 = 40
        #gridSize1 = 0.1
        #gridSize2 = 0.05
        #gridSize3 = 0.02

        #coords[coords >= lim2] = 400 + np.round((coords[coords>=lim2]) / gridSize1)
        #coords[coords < lim2] = np.round(coords[coords<lim2] / gridSize2)

        #coords -= coords.min(0, keepdims=1)

        feats = block 

        # sparse quantization: filter out duplicate points
        indices, inverse = sparse_quantize(coords,
                                        feats,
                                        return_index=True,
                                        return_invs=True)
        coords = coords[indices]
        feats = feats[indices]

        # construct the sparse tensor
        inputs = SparseTensor(feats, coords)
        inputs = sparse_collate_tensors([inputs]).to(cuda1)

        # make pred
        outputs = model(inputs)
        outputs = outputs.argmax(1).cpu().numpy()
        outputs = outputs[inverse]

        scanMatrix = np.zeros((19, 19))
        zMatrixs = np.zeros((16, 19, 19))
        xyMatrixs = np.zeros((11, 19, 19))

        if skan % 500 == 0:
            # save some outputs
            res = np.insert(lidar, 4, outputs, axis = 1)
            res.tofile(model_name + "/" + lidars[skan]) 

        for point in range(lidar.shape[0]):
            # count index
            zIndex = int( 2 * (lidar[point][2] + 4) ) # co 0.5m nowy przedział
            if zIndex < 0:
                zIndex = 0
            if zIndex > 15:
                zIndex = 15

            xyIndex = int( 2 * np.sqrt(lidar[point][0]**2 + lidar[point][1]**2) / 10)  # co 5 m nowy przedział
            if xyIndex > 9 :
                xyIndex = 10

            # add values to structs
            scanMatrix[outputs[point], label[point]] += 1
            zMatrixs[zIndex, outputs[point], label[point]] += 1
            xyMatrixs[xyIndex, outputs[point], label[point]] += 1

        # count general ious
        scanIou = np.zeros((19))
        scanAcc = np.zeros((19))
        for s in range(scanMatrix.shape[0]):
            scanIou[s] = countIou(scanMatrix, s)
            scanAcc[s] = countAcc(scanMatrix, s)
        generalIouArray[skan] = np.mean(scanIou[scanIou >= 0])
        generalAccArray[skan] = np.mean(scanAcc[scanAcc >= 0])
        # count iou for xy
        for dist in range(11):
            for kittiClass in range(19):
                xyIouArray[skan, dist, kittiClass] = countIou(xyMatrixs[dist], kittiClass)
                xyAccArray[skan, dist, kittiClass] = countAcc(xyMatrixs[dist], kittiClass)
        # cout iou for z
        for dist in range(16):
            for kittiClass in range(19):
                zIouArray[skan, dist, kittiClass] = countIou(zMatrixs[dist], kittiClass)
                zAccArray[skan, dist, kittiClass] = countAcc(zMatrixs[dist], kittiClass)

    # save all data
    generalIouArray.tofile(model_name + "/generaliou.bin")
    xyIouArray.tofile(model_name + "/xyiou.bin")
    zIouArray.tofile(model_name + "/ziou.bin")
    generalAccArray.tofile(model_name + "/generalacc.bin")
    xyAccArray.tofile(model_name + "/xyacc.bin")
    zAccArray.tofile(model_name + "/zacc.bin")


    endTime = time.time()
    print(model_name)
    print('finish!')
    print('start at ', time.ctime(startTime))
    print('end at ', time.ctime(endTime))
    print('general Iou: ', np.mean(generalIouArray[generalIouArray >= 0]))
    print('general Acc: ', np.mean(generalAccArray[generalAccArray >= 0]))
