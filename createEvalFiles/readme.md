This direcotry contain script for evaluate created model

all params are in consts.py

Script save eval files to eval/modelName,
    where modelName is name of current model

All important things are marked with "# TODO" 

Create data with structures:
    for iou metric
    General: np.matrix((scanNum)) - matrix contain mean iou for single scan
    XY: np.matrix((scanNum, xyBins, classes)) - each cell of matrix contain mean iou for single scan, single dist bin and single class
    Z: same as XY, but for zBins



Run script:

    python3 createEvalFiles.py
