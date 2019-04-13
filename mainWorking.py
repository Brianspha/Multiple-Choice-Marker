from pdf2image import convert_from_path
import os
from os import listdir
from os.path import isfile, join
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import imutils
import math
from operator import itemgetter
import time
import itertools
import shutil

#@dev reads in a pdf file and converts all pages into ppm files
#@param pdfFile name of pdf file
#@param outputname name of the output file/s
#@param directory directory to which the converted pdf pages are to be stored
def convertPdfToImages(pdfFile,outputname,directory,rotating=False,toRotate=[]):
    pages = convert_from_path(pdfFile,200)
    ###print("Opening File: ",pdfFile)
    index=1 
    if not os.path.exists(directory):
            shutil.rmtree(directory) 
            os.makedirs(directory)
    if not rotating:
            for page in pages:
                name=outputname+str(index)+".png"
                newfile=join(directory,name)
                page.save(newfile,'PNG')
                index+=1  
    else:
            for page in pages:
                name=outputname+str(index)+".png"
                #print(toRotate)
                newfile=join(directory,name)
                index+=1
                if name in toRotate:
                    #print("rotating",newfile)
                    page.save(newfile,'PNG')
                    cv2.imwrite(newfile,rotate_image(newfile,180))
                else:
                      page.save(newfile,'PNG')


#@dev filters folder and returns only ppm files
#@dev directoryName
def filterFolder(directoryName):
    onlyfiles = []
    for f in listdir(directoryName):
        cur =join(directoryName, f)
        split =os.path.splitext(cur)
        _,ext = split
        if isfile(cur) and ext==".png" :
            onlyfiles.append(f)
    return onlyfiles
#@dev converts bytes to int
#@param bytes- to be converted    
def bytes_to_int(bytes):
    result = 0
    for b in bytes:
        result = result * 256 + int(b)
    return result
#@dev removes b's from given number
def byteTostring(string):
    string=string.decode("utf-8")
    ####print("string", string)
    return  string  
from copy import copy, deepcopy
def rotate_image(file, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    imgfile=cv2.imread(file)
    height, width = imgfile.shape[:2] # image shape has 3 dimensions
    height1,width1=imgfile.shape[:2]
    image_centerRight=(width1/2,height/2)
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    
    rotate_imageRight=cv2.getRotationMatrix2D(image_centerRight, angle, 1.)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotate_imageRight[0,0]) 
    abs_sin = abs(rotate_imageRight[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_Image = cv2.warpAffine(imgfile, rotation_mat, (bound_w, bound_h))
    return rotated_Image
def rotate():
    files =filterFolder("ConvertedPages")
    index=1
    for file in files:
        imgFile = cv2.imread(join("ConvertedPages",file),1)
        imgFile=rotate_image(imgFile,90)
        cv2.imwrite(join("ConvertedPages",file),imgFile)
        index+=1
def extractInnerRectangles():
    files=filterFolder("ConvertedPages")
    toFix=[]
    for file in files:
        file = join("ConvertedPages",file)
        ###print("File",file)
        imgFile=cv2.imread(file,1)
        gray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_OTSU)
        contours,h = cv2.findContours(thresh,1,2)
        squares=[] 
        
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            
            ###print (len(approx))
            if len(approx)==4:
                x, y, width, height = cv2.boundingRect(approx)
                aspectRatio = float(width) / height

                # a square will have an aspect ratio that is approximately
                # equal to one, otherwise, the shape is a rectangle
                if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                    shape = "square"
                else:
                    shape = "rectangle"
                    hull = cv2.convexHull(cnt)
                    squares.append(hull)
                    cv2.drawContours(imgFile,[hull],0,(255,255,255),-1)
            
                ###print ("square")
                
        ###print("Found squres: ",shape)
        sort=sorted(squares, key=lambda x: cv2.contourArea(x))
        last4=[]
        if len(sort) ==0: #@dev if we read in no contours move on
            continue
        for i in range(len(sort)-2,len(sort)):
            last4.append(sort[i])
            cv2.drawContours(imgFile,sort[i],-1,(0,0,0),3)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(imgFile, cv2.MORPH_CLOSE,kernel, iterations=1)
        cv2.imwrite(file,closing)
        imgFile=cv2.imread(file,-1)
        grey = cv2.GaussianBlur(imgFile, (3,3), 0)
        grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(grey,127,255,cv2.THRESH_OTSU)
        contours,h = cv2.findContours(thresh,1,2)
        ###print("Found conts: ",len(contours))
        height, width = imgFile.shape[:2]
        blank_image = np.zeros((imgFile.shape[0],width,3), np.uint8)
        length =len(contours)
        for index in range(0,length-1):
            cv2.drawContours(blank_image,contours[index],-1,(255,255,0),3)
        cv2.imwrite(file,blank_image)
        imgFile=cv2.imread(file,1)
        imgFile = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgFile,220,255,cv2.THRESH_OTSU)
        kernel = np.ones((40, 40), np.uint8)
        thresh=cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
        cv2.imwrite(file,thresh)
        imgFile=cv2.imread(file,-1)
        grey = cv2.GaussianBlur(imgFile, (3,3), 0)
        ret,thresh = cv2.threshold(grey,127,255,cv2.THRESH_OTSU)
        contours,h = cv2.findContours(thresh,1,2)
        ##print(len(contours),"found")
        if (len(contours))>4 or len(contours)==0:
            toFix.append(file.split('\\')[1])
    return toFix
def extractCorners(toFix=[]):
    index=1
    files=filterFolder("ConvertedPages")
    toFix=[]
    for file in files:
        file = join("ConvertedPages",file)
        ####print("File",file)
        imgFile=cv2.imread(file,1)
        gray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        blank_image = np.zeros((height,width,3), np.uint8)
        ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_ERODE,kernel, iterations=9)
        erode = cv2.morphologyEx(closing, cv2.MORPH_CLOSE,kernel, iterations=15)
        cv2.imwrite(file,erode)
        contours,h = cv2.findContours(erode,1,2)
        ####print(len(contours))
        squares=[] 
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            area =cv2.contourArea(cnt)
            t=cv2.arcLength(cnt,True)
            ###print ("approx",len(approx))
            if area> 200 and area <2100:
                x, y, width, height = cv2.boundingRect(approx)
                aspectRatio = float(width) / height
                ####print("Here")
                # a square will have an aspect ratio that is approximately
                # equal to one, otherwise, the shape is a rectangle
                hull = cv2.convexHull(cnt)
                squares.append(hull)
                cv2.drawContours(blank_image,[hull],0,(255,255,255),-1)
                squares.append([cnt,x,y])
        cv2.imwrite(file,blank_image)
def extractInnerRectangle(file):
        file = join("ConvertedPages",file)
        ####print("File",file)
        imgFile=cv2.imread(file,1)
        gray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_OTSU)
        contours,h = cv2.findContours(thresh,1,2)
        squares=[] 
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            ####print (len(approx))
            if len(approx)==4:
                x, y, width, height = cv2.boundingRect(approx)
                aspectRatio = float(width) / height
                # a square will have an aspect ratio that is approximately
                # equal to one, otherwise, the shape is a rectangle
                if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                    shape = "square"
                else:
                    shape = "rectangle"
                    hull = cv2.convexHull(cnt)
                    squares.append(hull)
                    cv2.drawContours(imgFile,[hull],0,(255,255,255),-1)
            
                ####print ("square")
        ####print("Found squres: ",shape)
        #print(len(squares),"squares")
        sort=sorted(squares, key=lambda x: cv2.contourArea(x))
        last4=[]
        for i in range(len(sort)-5,len(sort)-3):
            last4.append(sort[i])
            cv2.drawContours(imgFile,sort[i],-1,(0,0,0),3)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(imgFile, cv2.MORPH_CLOSE,kernel, iterations=1)
        cv2.imwrite(file,closing)
        imgFile=cv2.imread(file,-1)
        grey = cv2.GaussianBlur(imgFile, (3,3), 0)
        grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(grey,127,255,cv2.THRESH_OTSU)
        contours,h = cv2.findContours(thresh,1,2)
        ####print("Found conts: ",len(contours))
        height, width = imgFile.shape[:2]
        blank_image = np.zeros((imgFile.shape[0],width,3), np.uint8)
        length =len(contours)
        for index in range(0,length-1):
            cv2.drawContours(blank_image,contours[index],-1,(255,255,0),3)
        cv2.imwrite(file,blank_image)
        imgFile=cv2.imread(file,1)
        imgFile = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgFile,220,255,cv2.THRESH_OTSU)
        kernel = np.ones((40, 40), np.uint8)
        thresh=cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
        cv2.imwrite(file,thresh)
        imgFile=cv2.imread(file,-1)
        grey = cv2.GaussianBlur(imgFile, (3,3), 0)
        ret,thresh = cv2.threshold(grey,127,255,cv2.THRESH_OTSU)
        contours,h = cv2.findContours(thresh,1,2)
        ###print(len(contours),"found")
def extractCorner(file,pdFfile,namingConvention):
    index=1
    pages = convert_from_path(pdFfile,200)
    index=1
    for page in pages:
        name=namingConvention+str(index)+".png"
        ###print("name",name,"File",file)
        if name in file: 
            file = join("ConvertedPages",file)
            page.save(file,'PNG')
            ###print("Found",name)
        index+=1
    ###print("File",file)
    imgFile=cv2.imread(file,1)
    gray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY)
    height,width=gray.shape
    blank_image = np.zeros((height,width,3), np.uint8)
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_ERODE,kernel, iterations=9)
    erode = cv2.morphologyEx(closing, cv2.MORPH_CLOSE,kernel, iterations=15)
    cv2.imwrite(file,erode)
    contours,h = cv2.findContours(erode,1,2)
    ####print(len(contours))
    squares=[] 
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        area =cv2.contourArea(cnt)
        t=cv2.arcLength(cnt,True)
        ###print ("approx",len(approx))
        if area> 200 and area <2500:
            x, y, width, height = cv2.boundingRect(approx)
            ####print("Here")
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            hull = cv2.convexHull(cnt)
            squares.append(hull)
            cv2.drawContours(blank_image,[hull],0,(255,255,255),-1)
            squares.append([cnt,x,y])
    cv2.imwrite(file,blank_image)# Getting the current work directory (cwd)
def extractPagesfromPDF(pages):
    files =filterFolder("ConvertedPages")
def verifyImageContours():
    files=filterFolder("ConvertedPages")
    cantMark=[]
    for file in files:
        file=join("ConvertedPages",file)
        imgfile=cv2.imread(file,0)
        conts,_=cv2.findContours(imgfile,1,2)
        count=len(conts)
        ##print(count,"File",file)
        if count >7: #@dev we setting the min number of contours we accept to 7
            cantMark.append(file.split('\\')[1])
    return cantMark
def Sort_Tuple(tup,by):  
      
    # getting length of list of tuples 
    lst = len(tup)  
    for i in range(0, lst):  
          
        for j in range(0, lst-i-1):  
            if (tup[j][by] > tup[j + 1][by]):  
                temp = tup[j]  
                tup[j]= tup[j + 1]  
                tup[j + 1]= temp  
    return tup  
def calculateAngleMinor(toFix=[]):
    files=filterFolder("ConvertedPages")
    cantMark=[]
    points=[]
    #XY=[]
    correct=[]
    incorrect=[]
    for file in files:
        if file in toFix:
            #print("Skipping")
            continue
        file=join("ConvertedPages",file)
        imgfile=cv2.imread(file)
        gray =cv2.imread(file,0)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        # otherwise, just take the inverse of the angle to make
        # it positive
        else:
            angle = -angle
        # rotate the image to deskew it
        (h, w) = imgfile.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(imgfile, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(file,imgfile)
        print("angle for file",file,angle)
    #print("Correct files: ",len(correct),"\n \n","Incorrect files: ",len(incorrect))
def createBlankImage(width,height):
    return np.zeros((height,width,3), np.uint8)
def drawLine(x,y,x1,y1,imgfile,iterations=1):
        for i in range(iterations):
                cv2.rectangle(imgfile,(x1,y),(x,y1),222,thickness=19,lineType=5)
        return imgfile
def getPDFpages(pdfFile):
    pages = convert_from_path(pdfFile)
    return pages
def calculateAngleMajor(toFix):
        files=filterFolder("ConvertedPages")
        cantMark=[]
        points=[]
        #XY=[]
        correct=[]
        incorrect=[]
        X=[]
        Y=[]
        ##print("lenfiles",(files))
        for file in files:
            if file in toFix:
                #print("Skipping")
                continue
            file=join("ConvertedPages",file)
            imgfile=cv2.imread(file,0)
            h,w=imgfile.shape            
            conts,_=cv2.findContours(imgfile,1,2)
            for contour in conts:
                approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
                x, y, width, height = cv2.boundingRect(approx)
                ###print("Found x and y",x,y)
                points.append((contour,(x,y)))
                ##print("conlen",len(contour))
                #XY.append((x,y))
                X.append(x)
                Y.append(y)
            #cls
            # #print("Before",XY)
            #XY=Sort_Tuple(XY,0)
            ##print("sorted",XY,"Current File",file,"\n \n \n")
            Xmin=min(X)
            Ymin=min(Y)
            Xmax=max(X)
            Ymax=max(Y)
            blank=createBlankImage(w,h)
            imgfile=drawLine(Xmin,Ymin,Xmax,Ymax,imgfile,1)
            radAngle=math.atan2(Xmax-Xmin,Ymax-Ymin)
            angle = math.degrees(radAngle)
            #print("angle for file",file,angle,"\n \n")
            cv2.imwrite(file,imgfile)
            if angle <37:
                incorrect.append(file)
            else:
                correct.append(file)
def rotateFiles(pdfFile,namingConvention):
    files =filterFolder("ConvertedPages")
    toRotate=[]
    start =time.time()
    for file in files:
        extractInnerRectangle(file)
        file=join("ConvertedPages",file)
        imgfile=cv2.imread(file,1)
        imgfile=cv2.cvtColor(imgfile,cv2.COLOR_BGR2GRAY)
        _,imgfile=cv2.threshold(imgfile,60,255,cv2.THRESH_OTSU)
        contours,_=cv2.findContours(imgfile,1,2)
        height,width=imgfile.shape
        centerX=int(width/2)
        centerY=int(height/2)
        #imgfile=imgfile[centerX,centerY]
        #print("centerX","cY",centerX,centerY)
        test=cv2.findNonZero(imgfile)
        #print(len(test),"file",file)
        blank=createBlankImage(width,height)
        centerY2=centerY/2
        centerX2=centerX/2
        for point in test:
            ##print(point)
            point=point[0]
            x,y=point
            if x>=centerX and y>= centerY2:
                if file.split("\\")[1] not in toRotate:
                    toRotate.append(file.split("\\")[1])
    #print(toRotate)
    convertPdfToImages(pdfFile,namingConvention,"ConvertedPages",True,toRotate)
    

thisdir = os.getcwd()
pdFfile =join(thisdir,"MCQ6002016.pdf")
naming="MCQ6002016"
convertPdfToImages(pdFfile,naming,"ConvertedPages")
#extractInnerRectangles()
#toFix=verifyImageContours()
#extractCorners(pdFfile)
#points =calculateAngleMinor(toFix)
#calculateAngleMajor(toFix)
##print("fix the ffg",toFix)
#file = ("MCQ60020162.png")
#extractInnerRectangle(file)
#imgFile=cv2.imread(file,1)
rotateFiles(pdFfile,naming)
#calculateAngleMinor()
extractInnerRectangles()