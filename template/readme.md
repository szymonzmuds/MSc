This template provides some scripts to visualize data

It is recommended to use this as template - just paste it, modify and work

This template require pandas, matplotlib and numpy


Data structures:
General: np.matrix((scanNum)) - matrix contain mean iou for single scan
XY: np.matrix((scanNum, xyBins, classes)) - each cell of matrix contain mean iou for single scan, single dist bin and single class
Z: same as XY, but for zBins


Describe:
all scripts do not require any params
run boxplot.py to get boxplots by dist and by classes and summary boxplot for all classes and all dist
this save results to directories: class, dist, partials

    python3 boxplot.py
    
run plotbydist.py to get plots by dist for all classes
this save results to directory: plotByDist
    
    python3 plotbydist.py
    
to modify anything check consts.py, where are all configs variables
