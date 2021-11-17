import numpy as np

from consts import *
from utils import *



def createGeneralBoxplots(models): 
    for dist in range(len(xyDist)):
        for kitti in range(len(kittiClasses)):
            dataDict = {}
            nameToSave = "partial/class_" + kittiClasses[kitti] + "_dist_" + str(xyDist[dist])
            for modelIndex in range(len(models)):
                dataDict[names[paths[modelIndex]]] = models[modelIndex][:, dist, kitti]
            createBoxplot(dataDict, nameToSave, "dist: "+str(xyDist[dist])+", class: "+kittiClasses[kitti])



def createBoxplotsForAllClasses(models):
    classIous = np.zeros((scanNum, len(kittiClasses), len(paths)))
    for scan in range(scanNum):
        for kitti in range(len(kittiClasses)):
            for model in range(len(paths)):
                mdl = models[model][scan, :, kitti]
                classIous[scan, kitti, model] = mdl[mdl >= 0].mean()

    for kitti in range(len(kittiClasses)):
        # save all classes to dict - it is easy to convert it to pandas dataframe later
        dataDict = {}
        nameToSave = "class/" + kittiClasses[kitti]
        title = "class: " + kittiClasses[kitti]
        for model in range(len(models)):
            dataDict[names[paths[model]]] = classIous[:, kitti, model]
        createBoxplot(dataDict, nameToSave, title)



def createBoxplotsForAllDists(models):
    distIous = np.zeros((scanNum, len(xyDist), len(paths)))
    for scan in range(scanNum):
        for dist in range(len(xyDist)):
            for model in range(len(paths)):
                mdl = models[model][scan, dist, :]
                distIous[scan, dist, model] = mdl[mdl >= 0].mean()

    for dist in range(len(xyDist)):
        # save all dists to dict - it is easy to convert it to pandas dataframe later
        dataDict = {}
        nameToSave = "dist/dist_" + str(xyDist[dist])
        title = "dist [m] : " + str(xyDist[dist])
        for model in range(len(models)):
            dataDict[names[paths[model]]] = distIous[:, dist, model]
        createBoxplot(dataDict, nameToSave, title)



if __name__ == "__main__":
    # all variables are avaible to modify in consts.py 
    print("read data")
    models = []
    for model in paths:
        models.append(readData(model, xyIouData))

    print("create boxplots for single class and for single dist")
    createGeneralBoxplots(models)


    print("create boxplots for single class for all distance")
    createBoxplotsForAllClasses(models)


    print("create boxplots for single distance for all classes")
    createBoxplotsForAllDists(models)   
    