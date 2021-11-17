from os import set_blocking
import numpy as np 
import matplotlib.pyplot as plt


def plotAll( tranSteps ):
  for i in range(0,5):
    y1 = []
    for nx in x:
      if i!=4:
        y1.append(linfunc(newAs[i + 1], newBs[i + 1], nx))
      else:
        y1.append(nx*endA + newBs[i+1])
    plt.plot(z, y1)

def linfunc(a, b, x):
  return a * x + b


def smoothTransition(DataIn, tranStart, tranDur, tranSteps, endA, newAs, newBs, breakX):
  FunctionOutput = 0
  sign = np.sign(DataIn)
  DataIn = np.absolute(DataIn)
  if DataIn < tranStart:
    FunctionOutput = DataIn
  elif DataIn > (tranStart + tranDur):
    FunctionOutput = DataIn * endA + newBs[len(newBs) - 1]
  else:
    idx = checkIndex( DataIn , breakX)
    if idx == tranSteps - 1:
      FunctionOutput = DataIn * endA + newBs[tranSteps]
    else:
        # if DataIn > breakX[idx] and DataIn < breakX[idx + 1]:
        FunctionOutput = DataIn * newAs[idx + 1] + newBs[idx + 1]
  return FunctionOutput * sign


def checkIndex( Datain ,breakX):
  outIndex = len(breakX) - 1
  for idx in range(0, len(breakX) - 1):
    if Datain >= breakX[idx] and Datain <= breakX[idx + 1]:
      outIndex = idx
  return outIndex

def findAll(tranSteps, tranDur, tranStart, endA, newAs, breakX):
  newBs = [0]
  breakY = [tranStart]
  for i in range(0, tranSteps):
    if i == tranSteps - 1:
      newB = breakY[i] - breakX[i] * endA
      newBs.append(newB)
      newY = endA * (tranStart + tranDur) + newBs[i + 1]
      breakY.append(newY)
    else:
      newB = breakY[i] - breakX[i] * newAs[i + 1]
      newBs.append(newB)
      newY = newAs[i + 1] * breakX[i + 1] + newBs[i + 1]
      breakY.append(newY)
  return newBs, breakY

def getCoeffs( tranStart = 30, tranDur = 20, tranSteps = 10, endA = 0.5):
  breakX = np.linspace(tranStart, (tranStart + tranDur) - (tranDur/tranSteps), num=tranSteps)
  newAs = np.linspace(1, endA + (1-endA)/tranSteps, num=tranSteps)
  newBs, breakY = findAll(tranSteps, tranDur, tranStart, endA, newAs, breakX)

  return newAs, newBs, breakX, breakY
