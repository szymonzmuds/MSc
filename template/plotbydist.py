import numpy as np

from consts import *
from utils import *



def createPlotsForSingleClasses(models):
    for kitti in range( len(kittiClasses) ):
        singleModel = np.zeros(( len(models), len(xyDist)))

        for modelIndex in range( len(models) ):
            for dist in range( len(xyDist) ):
                model = models[modelIndex]
                data = model[:, dist, kitti]
                data = np.mean( data[data>=0] ) * 100
                singleModel[modelIndex, dist] = data

        nameToSave = "plotsByDist/" + kittiClasses[kitti]
        createSimplePlot(model, nameToSave)



def createPlotForAllClasses(models):
    plotAllClasses = np.zeros((len(models), len(xyDist)))

    for modelIndex in range(len(models)):
        for dist in range(len(xyDist)):
            mdl = models[modelIndex]
            data = mdl[:,dist,:]
            data = np.mean(data[data>=0]) * 100
            plotAllClasses[modelIndex, dist] = data

    nameToSave = "allModels"
    createSimplePlot(plotAllClasses, nameToSave)



if __name__ == "__main__":
    print("create all required directories")
    createDirectories(simplePlotsDirs)


    print("read data")
    models = []
    for model in paths:
        models.append(readData(model, xyIouData))

    print("create plots for single class")
    createPlotsForSingleClasses(models)

    print("create plot for all classes")
    createPlotForAllClasses(models)