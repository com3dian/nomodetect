import cv2
from utils import *

def nomoPreprocessing(imgPath):
    img = cv2.imread(imgPath)
    img = cv2.copyMakeBorder(img, 100, 100, 100, 100, 
                             cv2.BORDER_CONSTANT, value=(255,255,255))
    img = 255 - cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    img2 = 255 - cv2.threshold(img, 45, 255, cv2.THRESH_TOZERO)[1]
    
    lsd = cv2.createLineSegmentDetector(sigma_scale = 0.3, n_bins = 1024, quant = 2.0)
    lines = lsd.detect(img2)[0] # Position 0 of the returned tuple are the detected lines
    lines = lineThresholdFunc(lines, 0.05)
    drawnImg = lsd.drawSegments(img2, lines)
    return img2, drawnImg, lines