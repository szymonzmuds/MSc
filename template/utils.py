import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from consts import *

def readData(path, name):
    matrix = np.fromfile(path + name)
    matrix = np.reshape(matrix, newShape[name])
    return matrix

def createBoxplot(struct, name_to_save, title):
    df = pd.DataFrame.from_dict(struct)
    filter = ((df["0.2"] >= 0) & (df["0.1"] >= 0) & (df["0.06"] >= 0) & (df["0.05"] >= 0) & (df["0.04"] >= 0) & (df["0.02"] >= 0) & (df["0.01"] >= 0))
    df = df.loc[filter]
    boxplot = df.boxplot()
    boxplot.set_xlabel("models")
    boxplot.set_ylabel("mIou")
    boxplot.set_title(title)
    plt.savefig(name_to_save+'.png')
    plt.cla()

def createSimplePlot(modelsData, nameToSave):
    for model in range(modelsData.shape[0]):
        singleModel = modelsData[model]
        plt.plot(xyDist, singleModel, colors[model], label = namesList[model])
    plt.legend()
    plt.grid()
    plt.xlabel("Dystans [m]", size = "large")
    plt.ylabel("mIoU [%]", size = "large")
    plt.savefig(nameToSave + "png")
    plt.cla()