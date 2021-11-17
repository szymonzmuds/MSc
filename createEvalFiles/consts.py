import numpy as np
import torch
from asymptote import asymptote as asp

model_name = "eval/modelName"

MODEL_PATH = "/path/to/model/modelName/checkpoints/max-iou-test.pt"
PATH_CONFIG = "configs/semantic_kitti/spvcnn/cr0p5.yaml"  # path to config file

# path to dataset
pathLidars = "../KITTI/dataset/sequences/08/velodyne/" 
pathLabels = "../KITTI/dataset/sequences/08/labels/"

LABEL_MAP = np.array([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 0, 1, 19,
                    19, 19, 2, 19, 19, 3, 19, 4, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 5, 6, 7, 19, 19, 19, 19, 19, 19,
                    19, 8, 19, 19, 19, 9, 19, 19, 19, 10, 11, 12, 13,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 19, 14, 15, 16, 19, 19, 19, 19, 19,
                    19, 19, 17, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19])

# TODO
tranStart = 30
tranDur = 10
tranSteps = 10
endA = 1.5
newAs, newBs, breakX, breakY = asp.getCoeffs( tranStart, tranDur, tranSteps, endA)
# voxelSize = np.array([0.05, 0.005, 0.05]) # modify here if another grid
mod = 20
divider = 2
voxelSize = 0.05
cuda = torch.device('cuda:0')
