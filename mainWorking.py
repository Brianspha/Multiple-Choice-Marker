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
from copy import copy, deepcopy

#@dev reads in a pdf file and converts all pages into ppm files
#@param pdfFile name of pdf file
#@param outputname name of the output file/s
#@param directory directory to which the converted pdf pages are to be stored
def convertPdfToImages(pdfFile,outputname,directory,rotating=False,toRotate=[]):
    pages = convert_from_path(pdfFile,200)
    ######print("Opening File: ",pdfFile)
    index=1 
    if not os.path.exists(directory):
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
                ####print(toRotate)
                newfile=join(directory,name)
                index+=1
                if name in toRotate:
                    ####print("rotating",newfile)
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
        ######print("File",file)
        imgFile=cv2.imread(file,1)
        gray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY)
        #ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_OTSU)
        blured = cv2.GaussianBlur(gray, (5, 5), 0)
        notwanted, thresh = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        #closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations=9)
        eroding = cv2.morphologyEx(thresh, cv2.MORPH_ERODE,kernel, iterations=15)
        contours, _ = cv2.findContours(eroding, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imwrite(file,eroding)
        contours,h = cv2.findContours(thresh,1,2)
        squares=[] 
        blank=createBlankImage(eroding.shape[1],eroding.shape[0])
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            if  len(approx)>=4 :
                
                area =cv2.contourArea(approx)
                if area >70000:
                    ###print("area",area,"pagearea",pageArea,file)
                    x, y, width, height = cv2.boundingRect(approx)
                    aspectRatio = float(width) / height
                    # a square will have an aspect ratio that is approximately
                    # equal to one, otherwise, the shape is a rectangle
                    shape = "rectangle"
                    hull = cv2.convexHull(cnt)
                    squares.append(hull)
                    cv2.drawContours(blank,[hull],0,(255,255,255),5)
        blank=cv2.GaussianBlur(blank,(25,25),0)
        #blank=cv2.cvtColor(blank,cv2.COLOR_BAYER_BG2GRAY)
        cv2.imwrite(file,blank)
        
    return toFix
def pointToInt(point):
    x,y=point
    return int(x),int(y)
def getnotZero(array):
    for num in array:
        if num >=50:
            return num
def findGreaterThan(array,number):
    for item in array:
        if item>number:
            return item
def findLessThan(array,number):
    length=len(array)-1
    for index in range(length):
        item=array[index]
        if item>number:
            return array[index-1]
def getCorners():
        files=filterFolder("ConvertedPages")
        for file in files:
            file=join("ConvertedPages",file)
            img=cv2.imread(file,0)
            gray = cv2.GaussianBlur(img, (5, 5), 0)
            kernel = np.ones((3, 3), np.uint8)
            ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_OTSU)
            closing = cv2.morphologyEx(thresh, cv2.MORPH_ERODE,kernel, iterations=9)
            erode = cv2.morphologyEx(closing, cv2.MORPH_CLOSE,kernel, iterations=15)
            cont,hierar=cv2.findContours(thresh,1,2)
            dst = cv2.cornerHarris(gray,5,3,0.04)
            ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
            dialted=cv2.dilate(dst,None)
            dst = np.uint8(dialted)
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(dialted,np.float32(centroids),(5,5),(-1,-1),criteria)
            blank=createBlankImage(erode.shape[1],erode.shape[0])
            ####print("cont len",len(corners),file)
            x=[]
            y=[]
            #sorted(corners)
            for corner in corners:
                    x.append(corner[0])
                    y.append(corner[1])
                    ###print(corner)
            x=sorted(x)
            y=sorted(y)
            x=np.array(x).astype(int)
            y=np.array(y).astype(int)
            xMin=findGreaterThan(x,38)
            yMin=findGreaterThan(y,148)
            xMax=int(max(x))
            yMax=max(y)
            if yMax>2000:
                yMax=1950
            if yMin<170 and yMin >=145:
                yMin=170
            if yMax >=1590 and yMax< 1640:
                yMax=1605
            ####print(corners)
            ##print(xMin,yMin,xMax,yMax,file)
            blank=gray[yMin:yMax,xMin:xMax]
            #blank=cv2.Canny(blank,127,200)
            cv2.imwrite(file,blank)
def haveToRotate(file):
        #file = join("ConvertedPages",file)
        imgFile =cv2.imread(file,0)
        template = cv2.imread(join("templates",'g.png'),0)
        w, h = template.shape[::-1]
        #print(w,h)
        res = cv2.matchTemplate(imgFile,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)
        y=[]
        for pt in zip(*loc[::-1]):
            y.append(pt[1] + h)
        y=(list(set(y)))
        if len(y)==0:
            return True
        else:
            return False
def extractCorner(file,pdFfile,namingConvention):
    index=1
    pages = convert_from_path(pdFfile,200)
    index=1
    for page in pages:
        name=namingConvention+str(index)+".png"
        ######print("name",name,"File",file)
        if name in file: 
            file = join("ConvertedPages",file)
            page.save(file,'PNG')
            ######print("Found",name)
        index+=1
    ######print("File",file)
    imgFile=cv2.imread(file,1)
    gray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY)
    height,width=gray.shape
    blank_image = np.zeros((height,width,3), np.uint8)
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_ERODE,kernel, iterations=9)
    erode = cv2.morphologyEx(closing, cv2.MORPH_CLOSE,kernel, iterations=15)
    #cv2.imwrite(file,erode)
    contours,h = cv2.findContours(erode,1,2)
    #######print(len(contours))
    squares=[] 
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        area =cv2.contourArea(cnt)
        t=cv2.arcLength(cnt,True)
        ######print ("approx",len(approx))
        if area> 200 and area <2500:
            x, y, width, height = cv2.boundingRect(approx)
            #######print("Here")
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            hull = cv2.convexHull(cnt)
            squares.append(hull)
            cv2.drawContours(blank_image,[hull],0,(255,255,255),-1)
            squares.append([cnt,x,y])
    cv2.imwrite(file,blank_image)# Getting the current work directory (cwd)
def createBlankImage(width,height):
    return np.zeros((height,width,3), np.uint8)
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
        #####print("lenfiles",(files))
        for file in files:
            if file in toFix:
                ####print("Skipping")
                continue
            file=join("ConvertedPages",file)
            imgfile=cv2.imread(file,0)
            h,w=imgfile.shape            
            conts,_=cv2.findContours(imgfile,1,2)
            for contour in conts:
                approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
                x, y, width, height = cv2.boundingRect(approx)
                ######print("Found x and y",x,y)
                points.append((contour,(x,y)))
                #####print("conlen",len(contour))
                #XY.append((x,y))
                X.append(x)
                Y.append(y)
            #cls
            # ####print("Before",XY)
            #XY=Sort_Tuple(XY,0)
            #####print("sorted",XY,"Current File",file,"\n \n \n")
            Xmin=min(X)
            Ymin=min(Y)
            Xmax=max(X)
            Ymax=max(Y)
            blank=createBlankImage(w,h)
            imgfile=drawLine(Xmin,Ymin,Xmax,Ymax,imgfile,1)
            radAngle=math.atan2(Xmax-Xmin,Ymax-Ymin)
            angle = math.degrees(radAngle)
            ####print("angle for file",file,angle,"\n \n")
            #cv2.imwrite(file,imgfile)
            if angle <37:
                incorrect.append(file)
            else:
                correct.append(file)
def rotateFiles(pdfFile,namingConvention):
    files =filterFolder("ConvertedPages")
    toRotate=[]
    start =time.time()
    for file in files:
        file=join("ConvertedPages",file)
        rotate=haveToRotate(file)
        if rotate:
            if file.split("\\")[1] not in toRotate:
                toRotate.append(file.split("\\")[1])
    ###print(toRotate)
    convertPdfToImages(pdfFile,namingConvention,"ConvertedPages",True,toRotate)
def getInnerRec(file):
        imgfile =cv2.imread(file,0)
        #imgGray=cv2.cvtColor(imgfile,cv2.COLOR_BGR2GRAY)
        #imgfile=getInnerRec(imgfile)
        #print(imgfile.shape)
        h,w=imgfile.shape
        ####print(h,w)
        startX=int(w*.4)
        startY=int(h*.004)
        EndX=int(w-w*.10)
        EndY=h-int(h*.015)
        box=imgfile[startY:EndY,startX:EndX]
        #cv2.imwrite("extracted.png",box)
        return box
def getSegmant(box, parts,isOne=True):
            h,w=box.shape
            EndY=round(h/parts)
            ###print("EndY",EndY)
            EndY=h-EndY
            ###print("h-EndY",EndY)
            origin=box
            box=box[0:EndY,0:w]
            ###print(box.shape)
            origin=origin[EndY:h,0:w]
            return (box,origin)
def getIndividualSegment(box,left):
        #cv2.imwrite("before.png",box)
        if left:
            h,w=box.shape
            EndX=int(w*.40)
            #print("EndX",EndX)
            box=box[0:h,0:EndX]
            #cv2.imwrite("afterL.png",box)
            return box
        else:
            h,w=box.shape
            StartX=int(w*.57)
            ##print("StartX",StartX,"w",w)
            box=box[0:h,StartX:w]
            #cv2.imwrite("afterR.png",box)
            return box
def getSegmantW(img,parts):
            h,w=img.shape
            EndX=round(w/parts)
            ###print("EndX",EndX)
            EndX=w-EndX
            ###print("h-EndX",EndX)
            origin=img
            img=img[0:h,0:EndX]
            ###print(img.shape)
            origin=origin[0:h,EndX:w]
            return (img,origin)
def removeExtraWhiteSpaceTop(img=None):
    if img is None:
        files =filterFolder("ConvertedPages")
        for file in files:
            file=join("ConvertedPages",file)
            img=cv2.imread(file,0)
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            ret,thresh = cv2.threshold(blurred,127,255,cv2.THRESH_OTSU)
            edged = cv2.Canny(thresh, 50, 100)
            contours,h = cv2.findContours(thresh,1,2)
            # find contours in the image and initialize the mask that will be
            # used to remove the bad contours
            cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            mask = np.ones(edged.shape[:2], dtype="uint8") * 255
            # loop over the contours
            xarr=[]
            yarr=[]
            for c in cnts:
                approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
                x, y, width, height = cv2.boundingRect(approx)
                ###print(x,y,file)
                xarr.append(x)
                yarr.append(y)
                # if the contour is bad, draw it on the mask
                #if is_contour_bad(c):
            xarr=sorted(xarr)
            yarr=sorted(i for i in yarr if i>=43)
            ###print(xarr,"\n",yarr,"\n")    
            minx=min(xarr)
            miny=min(yarr)
            maxx=max(xarr)
            maxy=max(yarr)
            img=img[miny:maxy,minx:maxx]
            # remove the contours from the image and show the resulting images
            #img= cv2.bitwise_and(img, img, mask=mask)
            cv2.imwrite(file,img)
    else:
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            ret,thresh = cv2.threshold(blurred,127,255,cv2.THRESH_OTSU)
            edged = cv2.Canny(thresh, 50, 100)
            contours,h = cv2.findContours(thresh,1,2)
            # find contours in the image and initialize the mask that will be
            # used to remove the bad contours
            cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            mask = np.ones(edged.shape[:2], dtype="uint8") * 255
            # loop over the contours
            xarr=[]
            yarr=[]
            for c in cnts:
                approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
                x, y, width, height = cv2.boundingRect(approx)
                ###print(x,y,file)
                xarr.append(x)
                yarr.append(y)
                # if the contour is bad, draw it on the mask
                #if is_contour_bad(c):
            xarr=sorted(xarr)
            yarr=sorted(i for i in yarr if i>=43)
            ###print(xarr,"\n",yarr,"\n")    
            minx=min(xarr)
            miny=min(yarr)
            maxx=max(xarr)
            maxy=max(yarr)
            
            return img
def getRowQuestion(image,left):
        side=getIndividualSegment(image,left)
       # cv2.imwrite("sideInRowQuest.png",side)
        #image=cv2.imread("detected1.png",0)
        template = cv2.imread(join("templates",'abc.png'),0)
        w, h = template.shape[::-1]
        #print(w,h)
        res = cv2.matchTemplate(side,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)
        y=[]
        for pt in zip(*loc[::-1]):
            y.append(pt[1] + h)
        if(len(y))==0:
            #reject sheet
            return [],False
        y=max(list(set(y)))

        h,w=side.shape
        side=side[y:h,0:w]
        #print(y,"Y")
        cv2.imwrite("matched1.png",side)
        h,w=side.shape
        maxY=38
        startX=40
        EndX=195
        part1=side[4:maxY,startX:EndX]
        h,w=part1.shape
        maxY1=int(maxY*2)
        part2=side[maxY:maxY1,startX:EndX]
        h,w=part2.shape
        maxY2=int(maxY*3)-5
        part3=side[maxY1:maxY2,startX:EndX]
        h,w=part3.shape
        maxY3=int(maxY*4)-8
        part4=side[maxY2:maxY3,startX:EndX]
        h,w=part4.shape
        maxY4=int(maxY*5)-12
        part5=side[maxY3:maxY4,startX:EndX]
        cv2.imwrite("part11.png",part1)
        cv2.imwrite("part12.png",part2)
        cv2.imwrite("part13.png",part3)
        cv2.imwrite("part14.png",part4)    
        cv2.imwrite("part15.png",part5)
        return [part1,part2,part3,part4,part5],True
def getCircles(img):
        #print("img",img.shape)
        cv2.imwrite("currentStrip.png",img)
        h,w=img.shape
        maxX=26
        circle1=img[0:h,0:maxX]
        circle2=img[0:h,maxX:56]
        circle3=img[0:h,56:86]
        circle4=img[0:h,86:118]
        circle5=img[0:h,118:w]
        cv2.imwrite("Circle1.png",circle1)
        cv2.imwrite("Circle2.png",circle2)
        cv2.imwrite("Circle3.png",circle3)
        cv2.imwrite("Circle4.png",circle4)
        cv2.imwrite("Circle5.png",circle5)
        return [circle1,circle2,circle3,circle4,circle5]
def getAnswersForStrip(img):
        cv2.imwrite("current.png",img)
        circles =getCircles(img)
        ans=[]
        chrs=["A","B","C","D","E"]
        index=0
        for currentCircle in circles:
                ret,thresh=cv2.threshold(currentCircle,150,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
                closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,(6,6),iterations=5)
                inverted=cv2.bitwise_not(closing)
                ##print(len(inverted))
                cv2.imwrite("invCircle"+str(index)+".png",inverted)
                nonZero=cv2.countNonZero(inverted)
                #print("invCircle"+str(index)+".png","Nonzero",nonZero,index)
                if nonZero >=240:
                    ans.append((chrs[index]))
                index+=1
        return ans
def getNextBigDiff(array,indexstart):
    for index in range(indexstart,len(array)-1):
        #print("diff",array[index+1][0]-array[index][0])
        if array[index+1][0]-array[index][0]>100:
            return array[index+1][0]
def detectDiffOne(array):
    length=len(array)
    newArr=[]
    for index in range(1,length-1):
        item=array[index]
        a=item[0]
        b=item[1]
        ans=b-a
        if ans >1 and ans<=10:
            new=[b,getNextBigDiff(array,index+1)]
            newArr.append(new)
        else:
            newArr.append(item)
    return newArr
def keepGreaterMinDiff(array):
    length=len(array)-1
    new=[]
    #print(array)
    new=[]
    for index in range(length):
        a=array[index]
        b=array[index+1]
        if b-a>=100:
            new.append(a)
    #new.append(array[len(array)-1])
    if len(new)==4:
        new.append(array[len(array)-1])
    return new
def getParts(file):
        file=join("ConvertedPages",file)
        img_rgb = getInnerRec(file)
        #img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(join("templates",'line.png'),0)
        w, h = template.shape[::-1]
        #print(w,h)
        res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)
        y=[]
        for pt in zip(*loc[::-1]):
            #cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            y.append(pt[1] + h)
        y=[num  for num in y if num>100 ]
        y=set(y)
        y=sorted(list(y))
        y=groupItems(y,2) 
        new=[]
        for item in y:
            if item[0]>0 and item[1]>0:
                    new.append(item[0])
                    new.append(item[1])
        y=sorted(new) 
        y=keepGreaterMinDiff(y)
        #y=detectDiffOne(y)
        #print("inside",y)
        h,w=img_rgb.shape
        #cv2.imwrite("matched.png",img_rgb)
        #print(y)
        startX=5
        endX=190
        part1=img_rgb[0:y[0],0:w]
        part2=img_rgb[y[0]:y[1],0:w]
        part3=img_rgb[y[1]:y[2],0:w]
        part4=img_rgb[y[2]:y[3],0:w]
        part5=img_rgb[y[3]:y[4],0:w]
        part6=img_rgb[y[4]:h,0:w]
        #cv2.imwrite('detected1.png',part1)
        #cv2.imwrite('detected2.png',part2)
        #cv2.imwrite('detected3.png',part3)
        #cv2.imwrite('detected4.png',part4)
        #cv2.imwrite('detected5.png',part5)
        #cv2.imwrite('detected6.png',part6)
        return [part1,part2,part3,part4,part5,part6]
def writeToFile(answers,sheet,stdNo,rejected):
    index=0
    if rejected:
        if not os.path.exists("Results"):
            os.makedirs("Results") 
        csv=open(join("Results","rejected"+str(sheet)+".csv"),"w")
        for ans in answers:
            if index < len(answers)-1:
                csv.write(str(ans))
                csv.write(",")
        return
    answers=sorted(answers,key=lambda x:x[0])
    #print(answers)
    file=join("Results","results"+str(sheet)+".csv")
    answerArr=[]
    if not os.path.exists("Results"):
        os.makedirs("Results")
    csv=open(file,"w")
    length=len(answers)-1
    index=0
    csv.write(str(stdNo))
    csv.write(",")
    for ans in answers:
            if index < len(answers)-1:
                csv.write(str(ans))
                csv.write(",")
        
    csv.close()
def groupItems(list,tupleLength):
    newList=[]
    length=len(list)
    #print(list,"len",length)
    tempList=[]
    if len(list) ==12:
            length=len(list)-tupleLength
            #print(list)
            for index in range(0,length,tupleLength):
                        newList.append((list[index],list[index+1]))
            newList.append([0,list[len(list)-1]])
    if len(list)%3==0:
            for i in range(0,length,3):
                    Max=max([list[i],list[i+1],list[i+2]])
                    Min=min([list[i],list[i+1],list[i+2]])
                    newList.append([Min,Max])
    else:
        for i in range(0,length,3):
            if i+2 <length:
                Max=max([list[i],list[i+1],list[i+2]])
                Min=min([list[i],list[i+1],list[i+2]])
                newList.append([Min,Max])
            else:
                newList.append([0,list[len(list)-1]])
                break
    #print("After: ",newList)
    return newList
def getstudentNumberStrips(file):
        file=join("ConvertedPages",file)
        img=cv2.imread(file,0)
        template = cv2.imread(join("templates",'g.png'),0)
        w, h = template.shape[::-1]
        #print(w,h)
        res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)
        y=[]
        x=[]
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            y.append(pt[1] + h)
            x.append(pt[0]+w)
        h,w=img.shape
        minY=max(sorted(set(y)))
        minX=min(sorted(set(x)))
        maxX=int(w*.30)
        maxY=int(h*.87)
        #print("minX",minX,"maxY",maxY,"maxX",maxX,x,y,w,h)
        img=img[minY:maxY,minX:maxX]
       # cv2.imwrite("matchedG.png",img)
        #student number part end
        #get student number parts
        h,w=img.shape
        sizeX=38
        part1 = img[0:int(h*.38),0:sizeX]
        #cv2.imwrite("part1.png",part1)
        part2 = img[0:int(h*.38),sizeX:sizeX*2]
        #cv2.imwrite("part2.png",part2)
        part3 = img[0:h,sizeX*2:sizeX*3]
        #cv2.imwrite("part3.png",part3)
        part4 = img[0:int(h*.38),sizeX*3:sizeX*4]
        #cv2.imwrite("part4.png",part4)
        part5 = img[0:int(h*.38),sizeX*4:sizeX*5]
        #cv2.imwrite("part5.png",part5)
        part6 = img[0:int(h*.38),sizeX*5:sizeX*6]
        #cv2.imwrite("part6.png",part6)
        part7 = img[0:int(h*.38),sizeX*6:sizeX*7]
        #cv2.imwrite("part7.png",part7)
        #get student number parts end
        return [part1,part2,part3,part4,part5,part6,part7]
def getCirclesFromStrip(part):
    h,w=part.shape
    sizeY=37
    h,w=part.shape
    sizeY=37
    circleArr=[part[0:sizeY,0:w],part[sizeY:sizeY*2,0:w]]
    for i in range(3,11):
        circleArr.append(part[sizeY*(i-1):sizeY*i,0:w])
    return circleArr
def getCirclesFromStripCharacter(part):
    h,w=part.shape
    sizeY=37
    index=3
    circleArr=[part[0:sizeY,0:w],part[sizeY:sizeY*2,0:w]]
    #cv2.imwrite("charcter1.png",part[0:sizeY,0:w])
    #cv2.imwrite("charcter2.png",part[sizeY:sizeY*2,0:w])
    for i in range(3,27):
        #cv2.imwrite("character"+str(index)+".png",part[sizeY*(i-1)+12:(sizeY*i)+5,0:w])
        circleArr.append(part[sizeY*(i-1)+12:(sizeY*i)+5,0:w])
        index+=1
    return circleArr
def processCircle(cirlces):
    index=0
    if len(cirlces)<11:
        for circle in cirlces:
                ret,thresh=cv2.threshold(circle,150,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
                closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,(6,6),iterations=5)
                inverted=cv2.bitwise_not(closing)
                nonBlack=cv2.findNonZero(inverted)
                #print(len(nonBlack),"lenNonBlack <10")
                #cv2.imwrite("CurrentCircle.png",circle)
                if len(nonBlack)>500:
                     break
                index+=1
    else:
        alphs=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
        for circle in cirlces:
            ret,thresh=cv2.threshold(circle,150,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
            closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,(6,6),iterations=5)
            inverted=cv2.bitwise_not(closing)
            nonBlack=cv2.findNonZero(inverted)
            #print(len(nonBlack),"lenNonBlack")
            if len(nonBlack)>500:
                index=alphs[index]
                break
            index+=1
    return index
def markSheets(file=None):
    files =filterFolder("ConvertedPages")
    answers=[]
    sheetNumber=1
    ##print(file is None)
    if file is None:
        for file in files:
            Parts=getParts(file)
            #StudentNoPart=getStudentNo(file)
            #print(len(Parts),"parts")
            leftQuestionCount=1
            rightQuestionCount=31
            for part in Parts:
                segmentLeft,fine =getRowQuestion(part,True)
                segmentRight,fine1 =getRowQuestion(part,False)
                #if not fine and not fine1:
                  #  writeToFile(answers,file,[],True)
                  #  break
                for question in segmentLeft:
                    ans=getAnswersForStrip(question)
                    answers.append((leftQuestionCount,ans))
                    print(ans,leftQuestionCount,file)    
                    leftQuestionCount+=1
                for question in segmentRight:
                    ans =getAnswersForStrip(question)
                    answers.append((rightQuestionCount,ans))
                    #print(ans,rightQuestionCount,file)
                    rightQuestionCount+=1
            studentNoStrips=getstudentNumberStrips(file)
            count=1
            stdNo=""
            for strip in studentNoStrips:
                circles=[]
                if count != 3:
                    circles=getCirclesFromStrip(strip)
                    val=processCircle(circles)
                    stdNo+=str(val)
                else:
                     cirlcesArr=getCirclesFromStripCharacter(strip)
                     val=processCircle(cirlcesArr)
                     stdNo+=str(val)
                count+=1
            writeToFile(answers,file,["student No: ",stdNo],False)
            sheetNumber+=1
    else:
        Parts=getParts(file)
        leftQuestionCount=1
        rightQuestionCount=31
        index=0
        #print(len(Parts),"parts")
        for part in Parts:
            segmentLeft,fine =getRowQuestion(part,True)
            segmentRight,fine1 =getRowQuestion(part,False)
            cv2.imwrite("part.png",part)
            #if not fine and not fine1:
            #    writeToFile(answers,file,[],True)
             #   break
            
            ##print("partleft",len(segmentLeft))
            ##print("partright",len(segmentRight))
            #cv2.imwrite("part" +str(index)+".png",part)
            #cv2.imwrite("segW.png",segmentRight[index]) 
            for question in segmentLeft:
                #cv2.imwrite("questionL"+str(index)+".png",question)
                ans=getAnswersForStrip(question)
                ##print(ans,"left")
                answers.append((leftQuestionCount,ans))
                #print(ans,leftQuestionCount,file)
                leftQuestionCount+=1
            for question in segmentRight:
                #cv2.imwrite("questionR"+str(index)+".png",question)
                ans =getAnswersForStrip(question)
                ##print(ans,"right")
                answers.append((rightQuestionCount,ans))
                #print(ans,rightQuestionCount,file)
                rightQuestionCount+=1
            index+=1
            studentNoStrips=getstudentNumberStrips(file)
            count=1
            stdNo=""
            for strip in studentNoStrips:
                circles=[]
                if count != 3:
                    circles=getCirclesFromStrip(strip)
                    val=processCircle(circles)
                    stdNo+=str(val)
                else:
                     cirlcesArr=getCirclesFromStripCharacter(strip)
                     val=processCircle(cirlcesArr)
                     stdNo+=str(val)
                count+=1
        writeToFile(answers,file,["student No: ",stdNo],False)

thisdir = os.getcwd()
pdFfile =join(thisdir,"MCQ2016.pdf")
naming="MCQ2016"
convertPdfToImages(pdFfile,naming,"ConvertedPages")
rotateFiles(pdFfile,naming)
getCorners()
removeExtraWhiteSpaceTop()
file=join("MCQ20162.png")
markSheets(file)
#test=[292, 294, 295, 513, 514, 731, 732, 734, 951, 953, 954, 1173, 1174, 1393, 0, 1394]
#test=keepGreaterMinDiff(test,100)
##print(test)
#one=
