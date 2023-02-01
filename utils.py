import cv2
import numpy as np
from collections import defaultdict
from kernels import *

def getFontSize(ocrResult):
    '''
    get the estimated font size from OCR result
    ------------------------
    ocrResult: a list of [[rectangle range], content: string, confidence]
    
    return: a minimum font size(pixel)
    '''
    fontSize = np.inf
    for result in ocrResult:
        p1, p2, p3, p4 = result[0]
        fontSize = min(fontSize, abs(p1[1] - p3[1]) + abs(p1[0] - p3[0]))
    
    return fontSize


def getFontHeight(ocrResult):
    fontSize = np.inf
    for result in ocrResult:
        p1, p2, p3, p4 = result[0]
        fontHeight = min(fontSize, abs(p1[1] - p3[1]), abs(p1[0] - p3[0]))
    
    return fontHeight


def conv(kernelFunc, kernelSize, middleIndex, recRange, image):
    '''
    convolution
    -------------------
    kernelFunc: function, one of the functions from kernels module
    kernelSize: integer, kernel size
    middleIndex: integer, middle index of the kernel
    recRange: list with length 4, the rectangle range to perform convolution operation 
    image: nd_array, the input image in nd_array form
    
    return: nd array with convolution result otherwise 
    '''
    ans = np.zeros_like(image)
    for pixelX in range(int(recRange[0]), int(recRange[1])+1):
        for pixelY in range(int(recRange[2]), int(recRange[3])+1):
            imgSl = image[(pixelY - middleIndex):(pixelY + middleIndex + 1), \
                          (pixelX - middleIndex):(pixelX + middleIndex + 1)]
            
            value = kernelFunc(kernelSize, middleIndex, imgSl)
            
            ans[pixelY, pixelX] = value
    return ans


def lineThresholdFunc(lines, tanThreshold):
    '''
    detect only horizontal lines and filter out others
    -----------------------
    lines: list, the deteted lines: [[x1, x2, y1, y2], [], ...]
    tanThreshold: positive value, 
                  the suggested value of tanThreshold is 0.05 or 0.1

    return: return only horizontal lines
    '''
    ans = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        if abs((y2-y1)/(x2-x1)) > tanThreshold:
            continue
        else:
            ans.append(line)
    return np.array(ans)


def lineSetGetRange(lineList):
    '''
    from line list get a rectangle range around those lines
    

    |----------------|  - borderUpper
    |                |
    |----------------|  - borderLower
    
    |                |
    borderLeft       borderRight
    ------------------------
    lineList: list, list of lines

    return: 
    '''
    
    borderLeft, borderRight, borderUpper, borderLower = lineList[0][0], lineList[0][2], lineList[0][1], lineList[0][3]
    for line in lineList:
        x1, y1, x2, y2 = line
        if min(x1, x2) < borderLeft:
            borderLeft = min(x1, x2)
        if max(x1, x2) > borderRight:
            borderRight = max(x1, x2)
        if min(y1, y2) < borderUpper:
            borderUpper = min(y1, y2)
        if max(y1, y2) > borderLower:
            borderLower = max(y1, y2)
    return borderLeft, borderRight, borderUpper, borderLower


def lines2bars(lineList, widthLimit, lengthLimit):
    # i implemented this part by manually write a naive 1d DBSCAN-like algorithm, 
    # anyone knows a better idea is welcomed to reach out
    '''
    lineSet: a list of all lines detected by cv2.lsd and filtered by lineThresholdFunc
    widthLimit: a limit of width
    lengthLimit: a limit of length

    return: a (default) dictionary which keys are the Y axis value and values are line segments
    '''
    bars = defaultdict(list)
    
    for line in lineList:
        x1, y1, x2, y2 = line[0]
        # to check if the line is flat or if it is too short
        if abs(y1 - y2) > widthLimit or abs(x2 - x1) < lengthLimit:
            continue
        
        if len(bars.keys()) == 0:
            bars[(y1+y2)*0.5].append(line[0])
        else:
            count = 0
            for key in bars.keys():
                keyCount = 0
                value = bars[key]
                # get all current line segments that has the same height
                
                for lineSeg in value:
                    if max(abs(y1 - lineSeg[1]), abs(y2 - lineSeg[3])) < widthLimit:
                        keyCount += 1
                        
                if keyCount > 0.5 * len(value):
                    bars[key].append(line[0])
                    count += 1
                    break
                
            if count == 0:
                bars[(y1+y2)*0.5].append(line[0])
                
    return bars


def plotOCR(result, img): 
    '''
    plot the OCR result on the input picture
    ----------------------
    result: the OCR result
    img: input image

    return 0
    '''
    for rectangle, content, confidence in result:
        top_left, bottom_right = (int(rectangle[0][0]), int(rectangle[0][1])), (int(rectangle[2][0]), int(rectangle[2][1]))
        cv2.rectangle(img, top_left,bottom_right, (0,255,0), 1)
    
    plt.figure(dpi=150)
    plt.imshow(img)
    return 0