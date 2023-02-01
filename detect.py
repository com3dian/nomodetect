from kernels import *
from utils import *
from collections import defaultdict
from copy import deepcopy
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def gravityClustering(valueList):
    '''
    gravity smoothing and then 
    '''
    positionGrav, rangeGravList = [], range(len(valueList))
    for i in rangeGravList:
        positionGrav.append(1/(1 + np.absolute(i - np.array(rangeGravList))))
    positionGrav = np.stack(positionGrav, axis=0)
    
    gravityMass = []
    for i in rangeGravList:
        # gravityMass.append(valueList * valueList[i]/np.max(valueList))
        gravityMass.append(50 * valueList/np.max(valueList))
    gravityMass = np.stack(gravityMass, axis=0)
    
    positionGrav *= gravityMass
    print('shape position: ', positionGrav.shape)
    
    return np.sum(positionGrav, axis = 1)


def maximaClustering(recRange, maximaImage, fontSize):
    '''
    naive 3-sigma algorithm
    '''
    
    allValue = np.sum(maximaImage[int(recRange[2]):(int(recRange[3])+1), \
               int(recRange[0]):(int(recRange[1])+1)], axis = 0)
    allValue = gravityClustering(allValue)
    
    threshold = min(np.mean(allValue) + 3 * np.std(allValue), np.percentile(allValue, 98))
    
    ans, _ = find_peaks(allValue, height = threshold, distance = fontSize)
    ans += int(recRange[0])
    return list(ans)


def findMaxima(fontSize, recRange, image):
    '''
    find all local maximas in a rectangle area on the image
    '''
    kernelSize = 2 * int(fontSize/2) + 1
    middleIndex = int(fontSize/2)
    recRange = recRange[0] - middleIndex, recRange[1] + middleIndex, recRange[2], recRange[3]
    
    convImage = conv(convKernelFunc, kernelSize, middleIndex, recRange, image)
    maximaImage = conv(maxKernel, 3, 1, recRange, convImage)
    minmaxImage = min1max(recRange, middleIndex, maximaImage)

    
    ans = maximaClustering(recRange, minmaxImage, fontSize)
    
    return ans


def merge(valueList, fontSize, ocrResult):
    '''
    naive 3 sigma algorithm to find the large difference of the 
    input list and merge values that has a small difference
    ----------------------------------
    example input/output:
    valueList: [1, 1.01, 0.99, 2, 2.01, 2.02]
    return: [1, 2.01]
    '''
    
    newValueList = np.sort(valueList)
    difference = [0]
    for i in range(1, len(valueList)):
        difference.append(newValueList[i] - newValueList[i-1])
    
    fontHeight = getFontHeight(ocrResult)
    seg = [i for i in range(len(newValueList)) if difference[i] > fontHeight]
    seg.insert(0, 0)
    seg.append(len(newValueList))
    ans = []
    for i in range(len(seg) - 1):
        ans.append(np.mean(newValueList[seg[i]:seg[i+1]]))
    return ans