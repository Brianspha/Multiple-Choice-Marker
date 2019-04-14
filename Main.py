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
    #####print("Opening File: ",pdfFile)
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
                ###print(toRotate)
                newfile=join(directory,name)
                index+=1
                if name in toRotate:
                    ###print("rotating",newfile)
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
        #####print("File",file)
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
                    ##print("area",area,"pagearea",pageArea,file)
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
            ###print("cont len",len(corners),file)
            x=[]
            y=[]
            #sorted(corners)
            for corner in corners:
                    x.append(corner[0])
                    y.append(corner[1])
                    ##print(corner)
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
            ###print(corners)
            #print(xMin,yMin,xMax,yMax,file)
            blank=gray[yMin:yMax,xMin:xMax]
            #blank=cv2.Canny(blank,127,200)
            cv2.imwrite(file,blank)
def extractInnerRectangle(file):
        file = join("ConvertedPages",file)
        ######print("File",file)
        imgFile=cv2.imread(file,1)
        gray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_OTSU)
        contours,h = cv2.findContours(thresh,1,2)
        squares=[] 
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            ######print (len(approx))
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
            
                ######print ("square")
        ######print("Found squres: ",shape)
        ###print(len(squares),"squares")
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
        ######print("Found conts: ",len(contours))
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
        #####print(len(contours),"found")
def extractCorner(file,pdFfile,namingConvention):
    index=1
    pages = convert_from_path(pdFfile,200)
    index=1
    for page in pages:
        name=namingConvention+str(index)+".png"
        #####print("name",name,"File",file)
        if name in file: 
            file = join("ConvertedPages",file)
            page.save(file,'PNG')
            #####print("Found",name)
        index+=1
    #####print("File",file)
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
    ######print(len(contours))
    squares=[] 
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        area =cv2.contourArea(cnt)
        t=cv2.arcLength(cnt,True)
        #####print ("approx",len(approx))
        if area> 200 and area <2500:
            x, y, width, height = cv2.boundingRect(approx)
            ######print("Here")
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            hull = cv2.convexHull(cnt)
            squares.append(hull)
            cv2.drawContours(blank_image,[hull],0,(255,255,255),-1)
            squares.append([cnt,x,y])
    cv2.imwrite(file,blank_image)# Getting the current work directory (cwd)
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
        ####print("lenfiles",(files))
        for file in files:
            if file in toFix:
                ###print("Skipping")
                continue
            file=join("ConvertedPages",file)
            imgfile=cv2.imread(file,0)
            h,w=imgfile.shape            
            conts,_=cv2.findContours(imgfile,1,2)
            for contour in conts:
                approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
                x, y, width, height = cv2.boundingRect(approx)
                #####print("Found x and y",x,y)
                points.append((contour,(x,y)))
                ####print("conlen",len(contour))
                #XY.append((x,y))
                X.append(x)
                Y.append(y)
            #cls
            # ###print("Before",XY)
            #XY=Sort_Tuple(XY,0)
            ####print("sorted",XY,"Current File",file,"\n \n \n")
            Xmin=min(X)
            Ymin=min(Y)
            Xmax=max(X)
            Ymax=max(Y)
            blank=createBlankImage(w,h)
            imgfile=drawLine(Xmin,Ymin,Xmax,Ymax,imgfile,1)
            radAngle=math.atan2(Xmax-Xmin,Ymax-Ymin)
            angle = math.degrees(radAngle)
            ###print("angle for file",file,angle,"\n \n")
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
        ###print("centerX","cY",centerX,centerY)
        test=cv2.findNonZero(imgfile)
        ###print(len(test),"file",file)
        blank=createBlankImage(width,height)
        centerY2=centerY/2
        centerX2=centerX/2
        for point in test:
            ####print(point)
            point=point[0]
            x,y=point
            if x>=centerX and y>= centerY2:
                if file.split("\\")[1] not in toRotate:
                    toRotate.append(file.split("\\")[1])
    ##print(toRotate)
    convertPdfToImages(pdfFile,namingConvention,"ConvertedPages",True,toRotate)
def getInnerRec(file):
        imgfile =cv2.imread(file)
        imgGray=cv2.cvtColor(imgfile,cv2.COLOR_BGR2GRAY)
        h,w=imgGray.shape
        ###print(h,w)
        startX=int(w*.4)
        startY=int(h*.004)
        EndX=int(w-w*.10)
        EndY=h-int(h*.015)
        box=imgGray[startY:EndY,startX:EndX]
        return box
def getSegmant(box, parts,isOne=True):
            h,w=box.shape
            EndY=round(h/parts)
            ##print("EndY",EndY)
            EndY=h-EndY
            ##print("h-EndY",EndY)
            origin=box
            box=box[0:EndY,0:w]
            ##print(box.shape)
            origin=origin[EndY:h,0:w]
            return (box,origin)
def getIndividualSegment(box,left):
        if left:
            h,w=box.shape
            EndX=int(w*.40)
            box=box[0:h,0:EndX]
            return box
        else:
            h,w=box.shape
            StartX=int(w*.60)
            box=box[0:h,StartX:w]
            return box
def getSegmantW(img,parts):
            h,w=img.shape
            EndX=round(w/parts)
            ##print("EndX",EndX)
            EndX=w-EndX
            ##print("h-EndX",EndX)
            origin=img
            img=img[0:h,0:EndX]
            ##print(img.shape)
            origin=origin[0:h,EndX:w]
            return (img,origin)
def removeExtraWhiteSpaceTop():
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
            ##print(x,y,file)
            xarr.append(x)
            yarr.append(y)
            # if the contour is bad, draw it on the mask
            #if is_contour_bad(c):
        xarr=sorted(xarr)
        yarr=sorted(i for i in yarr if i>=43)
        ##print(xarr,"\n",yarr,"\n")    
        minx=min(xarr)
        miny=min(yarr)
        maxx=max(xarr)
        maxy=max(yarr)
        img=img[miny:maxy,minx:maxx]
        # remove the contours from the image and show the resulting images
        #img= cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(file,img)
def getRowQuestion(image,left):
        side=getIndividualSegment(image,left)
        blur = cv2.GaussianBlur(side,(5,5),0)
        ret,thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #edges=cv2.Canny(thresh,127,200)
        closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,(3,3),iterations=5)
        h,w=closing.shape
        xmin = 31
        ymin=64
        print(w,h)
        w=w-int(w*.04)
        closing=closing[ymin:h,xmin:w]
        original=closing
        h,w=closing.shape
        innerroYmax=int(h/5)
        part1=closing[0:innerroYmax,0:w]
        h,w=part1.shape
        print(h,h+innerroYmax,w)
        part2=original[h:h+innerroYmax,0:w]
        h,w=part2.shape
        print(h,h+innerroYmax,w)
        part3=original[h:h+innerroYmax,0:w]
        h,w=part3.shape
        print(h,h+innerroYmax,w)
        part4=original[h:h+innerroYmax,0:w]
        h,w=part4.shape
        print(h,h+innerroYmax,w)
        part5=original[h:h+innerroYmax,0:w]
        cv2.imwrite("part1Q.png",part1)
        cv2.imwrite("part2Q.png",part2)
        cv2.imwrite("part3Q.png",part3)
        cv2.imwrite("part4Q.png",part4)
        cv2.imwrite("part5Q.png",part5)

        return [part1,part2,part3,part4,part5] 
def getCircles(img):
        h,w=img.shape
        startX=10
        endX=163
        img=img[0:h,startX:endX]
        img,circle1=getSegmantW(img,5)
        cv2.imwrite("circle1.png",circle1)
        img,circle2=getSegmantW(img,4)
        cv2.imwrite("circle2.png",circle2)
        img,circle3=getSegmantW(img,3)
        cv2.imwrite("circle3.png",circle3)
        img,circle4=getSegmantW(img,2)
        cv2.imwrite("circle4.png",circle4)
        img,circle5=getSegmantW(img,1)
        cv2.imwrite("circle5.png",circle5)
        return [circle5,circle4,circle3,circle2,circle1]
def getAnswersForStrip(img):
        circles =getCircles(img)
        ans=[]
        chrs=["A","B","C","D","E"]
        for i,currentCircle in enumerate(circles):
                inverted=cv2.bitwise_not(currentCircle)
        cv2.imwrite("invCircle"+str(i)+".png",inverted)
        nonZero=cv2.countNonZero(inverted)
        print("invCircle"+str(i)+".png","Nonzero",nonZero,i)
        if nonZero <650:
            ans.append((chrs[i]))
        return ans
def getParts(file):
        file=join("ConvertedPages",file)
        ###print(file)
        box=getInnerRec(file)
        h,w=box.shape
        ###print(h,w)
        #cv2.imwrite("extracted.png",box)
        #cv2.imwrite("box.png",box)
        box,part6=getSegmant(box,6)
        ##print("box6 shape",part6.shape)
        cv2.imwrite("remain6.png",box)
        cv2.imwrite("part6.png",part6)
        box,part5=getSegmant(box,5)
        ##print("box5 shape",part5.shape)
        cv2.imwrite("remain5.png",box)
        cv2.imwrite("part5.png",part5)
        box,part4=getSegmant(box,4)
        ##print("box4 shape",part4.shape)
        cv2.imwrite("remain4.png",box)
        cv2.imwrite("part4.png",part4)
        box,part3=getSegmant(box,3)
        ##print("box3 shape",part3.shape)
        cv2.imwrite("remain3.png",box)
        cv2.imwrite("part3.png",part3)
        box,part2=getSegmant(box,2)
        ##print("box2 shape",part2.shape)
        cv2.imwrite("remain2.png",box)
        cv2.imwrite("part2.png",part2)
        box,part1=getSegmant(box,1)
        ##print("box1",part1.shape)
        #cv2.imwrite("box1.png",box)
        cv2.imwrite("part1.png",part1)
        return [part1,part2,part3,part4,part5,part6]
def markSheets(file=None):
    files =filterFolder("ConvertedPages")
    answers=[]
    sheetNumber=1
    if file is None:
        for file in files:
            Parts=getParts(file)
            leftQuestionCount=1
            rightQuestionCount=31
            for part in Parts:
                segmentLeft =getRowQuestion(part,True)
                segmentRight =getRowQuestion(part,False)
                saveStrips(segmentLeft,segmentRight,file)
                for question in segmentLeft:
                    ans=getAnswersForStrip(question)
                    answers.append((leftQuestionCount,ans))
                    leftQuestionCount+=1
                for question in segmentRight:
                    ans =getAnswersForStrip(question)
                    answers.append((rightQuestionCount,ans))
                    rightQuestionCount+=1
            writeToFile(answers,sheetNumber)
            #sheetNumber+=1
            break
    else:
        Parts=getParts(file)
        leftQuestionCount=1
        rightQuestionCount=31
        for part in Parts:
            segmentLeft =getRowQuestion(part,True)
            segmentRight =getRowQuestion(part,False)
            saveStrips(segmentLeft,segmentRight,file)
            ans=getAnswersForStrip(segmentLeft[0])
            print(ans)
def writeToFile(answers,sheet):
    answers=sorted(answers,key=lambda x:x[0])
    file=join("Results","results"+str(sheet)+".csv")
    csv=""
    answerArr=[]
    if not os.path.exists("Results"):
        os.makedirs("Results")
        csv=open(file,"w")
        for num in range(1,61):
            #csv.write(str(num))
            answerArr.append((num,[]))
    else:
        csv=open(file,"a")
        csv.write("\n")
    for answer in answers:
        print(answer)
        index=1
        for ans in answer[1]:
            #csv.write(ans)
            idx=0
            for answr in answerArr:
                num,ansarr=answr
                if num == ans[0]:
                        ansarr.append(num)
                        answerArr[index]=(num,ansarr)
                idx+=1
    length=len(answers)
    index=0
    for ans in answers:
        if index <length:
            qno,alpharr=ans
            print(qno,alpharr)
            csv.write(str(qno))
            for alph in alpharr:
                csv.write(alph)
                csv.write(",")

    csv.close()
def saveStrips(left,right,file):
    lenght =len(left)
    for index in range(lenght):
        cv2.imwrite(str(index)+"stripleft"+file,left[index])
        cv2.imwrite(str(index)+"stripRight"+file,right[index])

thisdir = os.getcwd()
pdFfile =join(thisdir,"MCQ2016.pdf")
naming="MCQ2016"
convertPdfToImages(pdFfile,naming,"ConvertedPages")
rotateFiles(pdFfile,naming)
getCorners()
removeExtraWhiteSpaceTop()
file=join("MCQ20163.png")
markSheets(file)