from pdf2image import convert_from_path
import os
from os import listdir
from os.path import isfile, join
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import imutils
from random import randint
import pandas as pd

#@dev reads in a pdf file and converts all pages into ppm files
#@param pdfFile name of pdf file
#@param outputname name of the output file/s
#@param directory directory to which the converted pdf pages are to be stored
def convertPdfToImages(pdfFile,outputname,directory,rotating=False,toRotate=[]):
    pages = convert_from_path(pdfFile,200)
    #######print("Opening File: ",pdfFile)
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
                #####print(toRotate)
                newfile=join(directory,name)
                index+=1
                if name in toRotate:
                    #####print("rotating",newfile)
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
def findGreaterThan(array,number):
    for item in array:
        if item>number:
            return item
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
            #####print("cont len",len(corners),file)
            x=[]
            y=[]
            #sorted(corners)
            for corner in corners:
                    x.append(corner[0])
                    y.append(corner[1])
                    ####print(corner)
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
            #####print(corners)
            ###print(xMin,yMin,xMax,yMax,file)
            blank=gray[yMin:yMax,xMin:xMax]
            #blank=cv2.Canny(blank,127,200)
            cv2.imwrite(file,blank)
def haveToRotate(file):
        #file = join("ConvertedPages",file)
        imgFile =cv2.imread(file,0)
        template = cv2.imread(join("templates",'g.png'),0)
        w, h = template.shape[::-1]
        ##print(w,h)
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
def createBlankImage(width,height):
    return np.zeros((height,width,3), np.uint8)
def rotateFiles(pdfFile,namingConvention):
    files =filterFolder("ConvertedPages")
    toRotate=[]
    for file in files:
        file=join("ConvertedPages",file)
        rotate=haveToRotate(file)
        if rotate:
            if file.split("\\")[1] not in toRotate:
                toRotate.append(file.split("\\")[1])
    ####print(toRotate)
    convertPdfToImages(pdfFile,namingConvention,"ConvertedPages",True,toRotate)
def getInnerRec(file):
        imgfile =cv2.imread(file,0)
        #imgGray=cv2.cvtColor(imgfile,cv2.COLOR_BGR2GRAY)
        #imgfile=getInnerRec(imgfile)
        ##print(imgfile.shape)
        h,w=imgfile.shape
        #####print(h,w)
        startX=int(w*.4)
        startY=int(h*.004)
        EndX=int(w-w*.10)
        EndY=h-int(h*.015)
        box=imgfile[startY:EndY,startX:EndX]
        #cv2.imwrite("extracted.png",box)
        return box
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
                ####print(x,y,file)
                xarr.append(x)
                yarr.append(y)
                # if the contour is bad, draw it on the mask
                #if is_contour_bad(c):
            xarr=sorted(xarr)
            yarr=sorted(i for i in yarr if i>=43)
            ####print(xarr,"\n",yarr,"\n")    
            minx=min(xarr)
            miny=min(yarr)-5
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
                ####print(x,y,file)
                xarr.append(x)
                yarr.append(y)
                # if the contour is bad, draw it on the mask
                #if is_contour_bad(c):
            xarr=sorted(xarr)
            yarr=sorted(i for i in yarr if i>=43)
            ####print(xarr,"\n",yarr,"\n")    
            minx=min(xarr)
            miny=min(yarr)
            maxx=max(xarr)
            maxy=max(yarr)
            
            return img
def getCircles(img):
        ##print("img",img.shape)
        h,w=img.shape
        maxX=30
        circle1=img[0:h,0:maxX]
        circle2=img[0:h,maxX:maxX*2]
        circle3=img[0:h,maxX*2:maxX*3]
        circle4=img[0:h,maxX*3:maxX*4]
        circle5=img[0:h,maxX*4:maxX*5]
        cv2.imwrite("Circle1.png",circle1)
        cv2.imwrite("Circle2.png",circle2)
        cv2.imwrite("Circle3.png",circle3)
        cv2.imwrite("Circle4.png",circle4)
        cv2.imwrite("Circle5.png",circle5)
        return [circle1,circle2,circle3,circle4,circle5]
def getAnswersForStrip(img):
        circles =getCircles(img)
        ans=[]
        chrs=["A","B","C","D","E"]
        index=0
        for currentCircle in circles:
                ret,thresh=cv2.threshold(currentCircle,350,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
                closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,(6,6),iterations=5)
                inverted=cv2.bitwise_not(closing)
                ###print(len(inverted))
               #cv2.imwrite("invCircle"+str(index)+".png",inverted)
                nonZero=cv2.countNonZero(inverted)
                #print(nonZero,index)
                if nonZero >=300:
                    ans.append((chrs[index]))
                index+=1
        return ans
def writeToFile(answers,sheet,stdNo,rejected,answersFile=None,isNegativeMarking=False):
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
    #print("before","\n",answers)
    answers=sorted(answers,key=lambda x:x[0])
    #print("After","\n",answers)
    file=join("Results","results"+str(sheet)+".csv")
    answerArr=[]
    if not os.path.exists("Results"):
        os.makedirs("Results")
    csv=open(file,"w")
    length=len(answers)-1
    index=0
    csv.write("[studentNo: "+stdNo+"]")
    csv.write(",")
    for ans in answers:
#            if index < len(answers)-1:
                csv.write(str(ans))
    total=getTotal(answers,isNegativeMarking,answersFile)
    csv.write("Total: "+str(total))
    marks=join("Results","marks.csv")
    if not os.path.isfile(marks):
        marks=open(marks,"w")
        marks.write("studentNo,")
        marks.write("total mark,")
        marks.write("\n")
        marks.write(stdNo+",")
        marks.write(str(total))
        marks.write("\n")
    else:
        marks=open(marks,"a")
        marks.write(stdNo+",")
        marks.write(str(total))
        marks.write("\n")
    csv.close()
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
        ##print("minX",minX,"maxY",maxY,"maxX",maxX,x,y,w,h)
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
                ##print(len(nonBlack),"lenNonBlack <10")
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
            ##print(len(nonBlack),"lenNonBlack")
            if len(nonBlack)>500:
                index=alphs[index]
                break
            index+=1
    if index ==10:
        index="?"
    if index ==26:
        index="?"
    return index
def getlessThan(array):
    diff = [array[i+1]-array[i] for i in range(len(array)-1)]
    avg = sum(diff) / len(diff)

    m = [[array[0]]]

    for x in array[1:]:
        if x - m[-1][-1] < avg:
            m[-1].append(x)
        else:
            m.append([x])
    return m
def get12PartsforSheet(file):
        side=getInnerRec(file)
        template = cv2.imread(join("templates",'abc.png'),0)
        w, h = template.shape[::-1]
        ##print(w,h)
        res = cv2.matchTemplate(side,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)
        y=[]
        x=[]
        for pt in zip(*loc[::-1]):
            cv2.rectangle(side, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            y.append(pt[1] + h)
            x.append(pt[0])
            x.append(pt[0]+w)

        if len(y)==0:
            return [],True
        y=sorted(set(y))
        y=getlessThan(y)
        x=sorted(set(x))
        x=getlessThan(x)
        #print(x)
        #print("\n")
        #print(y)
        minX=min(x[0])
        minY=min(y[0])
        maxX=max(x[1])
        maxY=max(y[0])+166
        #cv2.rectangle(side,(minX,minY),(maxX,maxY),(22,222,234))
        part1=side[minY:maxY,minX:maxX]

        minY=min(y[1])
        maxY=max(y[1])+166
        #cv2.rectangle(side,(minX,minY),(maxX,maxY),(22,222,234))
        part2=side[minY:maxY,minX:maxX]

        minY=min(y[2])
        maxY=max(y[2])+166
        #cv2.rectangle(side,(minX,minY),(maxX,maxY),(22,222,234))
        part3=side[minY:maxY,minX:maxX]

        minY=min(y[3])
        maxY=max(y[3])+166
        #cv2.rectangle(side,(minX,minY),(maxX,maxY),(22,222,234))
        part4=side[minY:maxY,minX:maxX]

        minY=min(y[4])
        maxY=max(y[4])+166
        #cv2.rectangle(side,(minX,minY),(maxX,maxY),(22,222,234))
        part5=side[minY:maxY,minX:maxX]

        minY=min(y[5])
        maxY=max(y[5])+166
        #cv2.rectangle(side,(minX,minY),(maxX,maxY),(22,222,234))
        part6=side[minY:maxY,minX:maxX]

        minX=min(x[2])
        minY=min(y[0])
        maxX=max(x[3])
        maxY=max(y[0])+166
        #cv2.rectangle(side,(minX,minY),(maxX,maxY),(22,222,234))
        part7=side[minY:maxY,minX:maxX]

        minY=min(y[1])
        maxY=max(y[1])+166
        #cv2.rectangle(side,(minX,minY),(maxX,maxY),(22,222,234))
        part8=side[minY:maxY,minX:maxX]

        minY=min(y[2])
        maxY=max(y[2])+166
        #cv2.rectangle(side,(minX,minY),(maxX,maxY),(22,222,234))
        part9=side[minY:maxY,minX:maxX]

        minY=min(y[3])
        maxY=max(y[3])+166
        #cv2.rectangle(side,(minX,minY),(maxX,maxY),(22,222,234))
        part10=side[minY:maxY,minX:maxX]

        minY=min(y[4])
        maxY=max(y[4])+166
        #cv2.rectangle(side,(minX,minY),(maxX,maxY),(22,222,234))
        part11=side[minY:maxY,minX:maxX]

        minY=min(y[5])
        maxY=max(y[5])+166
        #cv2.rectangle(side,(minX,minY),(maxX,maxY),(22,222,234))
        part12=side[minY:maxY,minX:maxX]

        #cv2.imwrite("matched.png",side)
        return [part1,part2,part3,part4,part5,part6,part7,part8,part9,part10,part11,part12],False
def getSegmnantAnswerStrips(side):
        h,w=side.shape
        ##print(y,"Y")
        #cv2.imwrite("matched1.png",side)
        h,w=side.shape
        maxY=34
        startX=0
        EndX=w
        part1=side[4:maxY,startX:EndX]
        h,w=part1.shape
        maxY1=int(maxY*2)
        part2=side[maxY:maxY1,startX:EndX]
        h,w=part2.shape
        maxY2=int(maxY*3)
        part3=side[maxY1:maxY2,startX:EndX]
        h,w=part3.shape
        maxY3=int(maxY*4)
        part4=side[maxY2:maxY3,startX:EndX]
        h,w=part4.shape
        maxY4=int(maxY*5)
        part5=side[maxY3:maxY4,startX:EndX]
        cv2.imwrite("part1.png",part1)
        cv2.imwrite("part2.png",part2)
        cv2.imwrite("part3.png",part3)
        cv2.imwrite("part4.png",part4)
        cv2.imwrite("part5.png",part5)
        return [part1,part2,part3,part4,part5]
def getStudentNumber(file):
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
    return stdNo
def markSheets(file,answersFile,NegativeMarking=False):
    files =filterFolder("ConvertedPages")
    answers=[]
    sheetNumber=1
    count=1
    leftQuestionCount=1
    rightQuestionCount=31
    if file:
        for file in files:
            stdNo=getStudentNumber(file)
            file=join("ConvertedPages",file)
            Parts,rejected=get12PartsforSheet(file)
            answers=[]
            if rejected:
                writeToFile(answers,sheetNumber,stdNo,rejected)
                continue
            else:   
                leftTurn=True
                qCount=1
                for part in Parts:
                    parts=getSegmnantAnswerStrips(part)
                    for strip in parts:
                        ans=getAnswersForStrip(strip)
                        answers.append([qCount,ans])  
                        qCount+=1
                file=file.split("\\")[1]
                writeToFile(answers,file,stdNo,False,answersFile,NegativeMarking)
    else:
        stdNo=getStudentNumber(file)
        file=join("ConvertedPages",file)
        Parts,rejected=get12PartsforSheet(file)
        if rejected:
            writeToFile([],sheetNumber,stdNo,rejected,NegativeMarking)
            return
        else:   
            leftTurn=True
            qCount=1
            for part in Parts:
                parts=getSegmnantAnswerStrips(part)
                for strip in parts:
                    ans=getAnswersForStrip(strip)
                    answers.append([qCount,ans])  
                    qCount+=1
            file=file.split("\\")[1]
            writeToFile(answers,file,stdNo,False,answersFile,NegativeMarking)
def getTotal(answers,NegativeMarking,answersFile):
    thisdir = os.getcwd()
    file=join(thisdir,answersFile)
    ans =open(file)
    ans=ans.readlines()[0].split('--')
    marksoFar=0
    negativeMark=0
    for index in range(len(answers)):
        studentAns=ans[index]
        correctans=answers[index]
        found=False
        correctans=correctans[1]
        for answr in correctans:
            if answr in studentAns:
                marksoFar+=1
                found=True
            else:
                negativeMark+=1
                found=False
    if NegativeMarking:
        marksoFar-=negativeMark
    return marksoFar
def analytics():
    if not os.path.isfile(join("Results","marks.csv")):
        raise Exception("No sheets were marked")
    else:
        marks=open(join("Results","marks.csv"))
        marks=marks.readlines()
        #print(marks)
        marks=marks[1:len(marks)]
        studentNos = []
        newMarks=[]
        for student in marks:
            newMarks.append(student.split(',')[1])
            studentNos.append(student.split(',')[0])
        updateMarks=[]
        for num in newMarks:
            if int(num) <=0:
                updateMarks.append(0)
            else:
                updateMarks.append(int(num))
        index = np.arange(len(studentNos))
        plt.bar(index, updateMarks)
        plt.xlabel('Student', fontsize=10)
        plt.ylabel('Mark', fontsize=10)
        plt.xticks(index, studentNos, fontsize=5, rotation=30)
        plt.title('Results out of 60 marks')
        plt.show()
def main():
    thisdir = os.getcwd()
    fileName =input('Please enter File name\n')
    pdFfile=""
    while(not fileName):
        fileName =input('Please enter File name\n')
    try:
        pdFfile =join(thisdir,fileName)
        ##print(os.path.isfile(join(thisdir,pdFfile)))
        if not os.path.isfile(join(thisdir,pdFfile)):
                 raise Exception("")
        else:
            try:
                answers=input("Enter answers File name: \n")
                while not answers:
                    answers=input("Enter answers File name: \n")
                answers=join(thisdir,answers)
               # #print(answers)
                #print(os.path.isfile(answers))
                if not os.path.isfile(answers):
                #    #print("Hello")
                    raise Exception("")
                else:
                    negativeMarking=input("Enable negative marking : Y/N \n")
                    if not negativeMarking :
                            negativeMarking=False
                    else:
                        negativeMarking=True
                    naming=fileName
                    #print("PDF file "+pdFfile)
                    #pages = convert_from_path(pdFfile,200)
                    convertPdfToImages(pdFfile,naming,"ConvertedPages")
                    rotateFiles(pdFfile,naming)
                    getCorners()
                    removeExtraWhiteSpaceTop()
                    markSheets(pdFfile,answers,negativeMarking)
                    analytics()
            except Exception as error:
                    print(answers +" does not exist")

    except:
            print(fileName," invalid file or file does not exist")

 
main() #to complete add abilty to plot graph add sys args add answers sheet option negative enabled
