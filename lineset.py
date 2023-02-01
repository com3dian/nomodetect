import cv2
from collections import defaultdict
from utils import *
from kernels import *
from detect import *

class lineSet:
    '''
    this class read a set of line segments and return the serrations
    '''
    def __init__(self, 
                 lineList, 
                 image, 
                 ocrResult,
                ):
        self.lineList = lineList
        self.endPointLeft, self.endPointRight = lineSetGetRange(self.lineList)[0], lineSetGetRange(self.lineList)[1]
        self.image = image
        self.ocrResult = ocrResult
        self.segment = self.getSegment(self.lineList, self.image, self.ocrResult)
        
    
    def getSegment(self, lineList, image, ocrResult):
        '''
        lineList: a list of lines within a height limit range
        image: opencv image object
        ocrResult: a list of 
                   [[rectangle range], content: string, confidence]
        -----------------------------
        return: a default dict which key is the y index and values is 
        a list of line segment end points.
        '''
        
        # get minimum font size
        fontSize = getFontSize(ocrResult)
        threshold = fontSize/4
        candidateList = []
        for lineIndex in range(len(lineList)):
            line = lineList[lineIndex]
            x1, y1, x2, y2 = line
            for anotherLine in lineList[lineIndex:]:
                x3, y3, x4, y4 = anotherLine
                if abs(x1 - x4) < threshold:
                    candidateList.append((x1+x4)*0.5)
                if abs(x2 - x3) < threshold:
                    candidateList.append((x2+x3)*0.5)
        
        candidateList += [self.endPointLeft, self.endPointRight]
        recRange = lineSetGetRange(self.lineList)
        maximaList = findMaxima(fontSize, recRange, image)
        
        ansList = candidateList
        for point in maximaList:
            for ansPoint in ansList:
                if abs(ansPoint - point) < fontSize/4 :
                    ansList.remove(ansPoint)
                    ansList.append((ansPoint + point)/2)
                    break
                else:
                    continue
                    
            ansList.append(point)
            
        ansKey = int(0.5 * (recRange[2] + recRange[3]))
        ans = defaultdict(list)
        ans[ansKey] = merge(ansList, fontSize, ocrResult)
        
        for scale in ans[ansKey]:
            top_left, bottom_right = (int(scale) - 10, int(ansKey) - 10), \
                                     (int(scale) + 10, int(ansKey) + 10)
            cv2.rectangle(image, top_left, bottom_right, 240, 1)
            
            plt.figure(dpi=150)
        
        plt.imshow(image)
        
        return ans