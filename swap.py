#written by Jeremy Lee (jalee)
#15-112 term project at Carnegie Mellon
import unsafeEventBasedAnimation #modified per prof kosbie's instructions to 
                                 #accommodate saving the canvas
import tkFileDialog
import os
from PIL import ImageTk, ImageDraw, ImageEnhance
import PIL.Image
from Tkinter import *
import shutil
import cv2 #allows for face recogition
import numpy as np
import math
import zipfile
import tkMessageBox
import tkColorChooser

#creates the face object, which allows an image to be processed to have faces
#detected and croppted
class Face(object):
    def __init__(self, img_path, width, height):
        self.fpos = []
        self.blur = 10
        self.width = width
        self.height = height
        self.img_path = img_path
        self.base = os.path.basename(self.img_path)
        self.dir = os.path.splitext(self.base)[0]
        #attempts to check if the directory exists, and removes it if it does
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.mkdir(self.dir)
        #provides paths to the cascades that allow for eye, mouth and 
        #face detection
        cascade_face_path = "haarcascade_frontalface_default.xml"
        cascade_eye_path = "haarcascade_eye.xml"
        cascade_mouth_path = "haarcascade_mouth.xml"
        #creates cascades for each facial feature
        cascade_face = cv2.CascadeClassifier(cascade_face_path)
        self.cascade_eye = cv2.CascadeClassifier(cascade_eye_path)
        self.cascade_mouth = cv2.CascadeClassifier(cascade_mouth_path)
        #opens image, and adds transparent layer
        self.image = cv2.imread(img_path, -1)
        #creates a list of coordinates of each face in an image
        self.detections_face = self.cascade_detect(cascade_face, self.image)
        self.detections_pt()

    #provides infrastructure to recognize features based on a cascade and img
    def cascade_detect(self, cascade, image):
        #provides minimum requirements for a feature to be recognized
        scale = 1.1
        neighbors = 7
        size = 10
        #converts image to greyscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cascade.detectMultiScale(
            grayscale,
            scaleFactor = scale,
            minNeighbors = neighbors,
            minSize = (size, size),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        
    #detects positions of the mouth within a face    
    def detect_mouth(self, image_cut, face_pt, (x,y,w,h)):
        detections_mouth = self.cascade_detect(self.cascade_mouth, image_cut)
        #checks to determine if a mouth is detected on a face
        if len(detections_mouth) > 0:
            pot_mouths = []
            #adds the mouth to the list of points
            for mouth in xrange(len(detections_mouth)):
                pot_mouths.append(detections_mouth[mouth][1])
            i = pot_mouths.index(max(pot_mouths))
            #records the corner coordinates of each mouth
            (xm,ym,wm,hm) = (detections_mouth[i][0], 
                            detections_mouth[i][1], 
                            detections_mouth[i][2], 
                            detections_mouth[i][3])
            face_pt.append((xm,ym,wm,hm))
            #draws the bounding box on the cropped face
            (x,y,w,h,f) = (x, y, w, h, self.detections_draw(face_pt, image_cut))   
            return (x,y,w,h,f)
        else:
            return None
        
    #detects eyes on each of the faces
    def detect_eye(self, image_cut, face_pt, (x,y,w,h)):
        detections_eye = self.cascade_detect(self.cascade_eye, image_cut)
        #ensures that there are atleast two eyes on each face
        if len(detections_eye) >= 2:
            for eye in xrange(2):
                #determines each corner point for each eye
                (xe,ye,we,he) = (detections_eye[eye][0], 
                                 detections_eye[eye][1], 
                                 detections_eye[eye][2], 
                                 detections_eye[eye][3])
                face_pt.append((xe,ye,we,he))
            #determines if the face also has a mouth
            mouth = self.detect_mouth(image_cut, face_pt, (x,y,w,h))
            if mouth != None:
                return mouth
            else:
                return None
     
    #finds the points containing the face
    def detections_pt(self):
        #process the picture and identify each point that contains a face
        for r in xrange(len(self.detections_face)):
            face_pt = []
            found = self.detections_face[r]
            #determines the height and width and location of face
            (x,y,w,h) = (found[0], found[1], found[2], found[3])
            #creates a output path for each cropped image in a folder with
            #same name as the image
            self.result_path = self.dir + "/" + str(r) + ".png"
            image = cv2.imread(self.img_path, -1)
            #crops each image to only contain the face
            image_cut = image[y:y+h, x:x+w]
            #checks for eyes in the image
            found = self.detect_eye(image_cut, face_pt, (x,y,w,h))
            #adds the face if there are two eyes and a mouth
            if found != None:
                self.fpos.append(found)
     
    #orders the eyes such that the left eye is on the left and right on the 
    #right
    def orderpt(self, face_pt):
        #checks x point and puts the two items in order
        if face_pt[0][0] > face_pt[1][0]:
            return [face_pt[1]] + [face_pt[0]] + [face_pt[2]]
        return face_pt
     
    #uses the points of each cropped face to determine position of each face 
    #on the non cropped image
    def findpt(self, face_pt):
        face_pt = self.orderpt(face_pt)
        #adds some padding to the points to capture a larger area of face
        scale = 4
        xmult = int(face_pt[0][2] / scale)
        #captures each corner of the image
        nw = ((face_pt[0][0] - xmult), face_pt[0][1])
        ne = ((face_pt[1][0]+(face_pt[1][2]) + xmult), face_pt[1][1])
        sw = ((face_pt[2][0]), face_pt[2][1] + (face_pt[2][3] / 2))
        se = ((face_pt[2][0]+(face_pt[2][2])), 
               face_pt[2][1] + (face_pt[2][3] / 2))    
        return (nw, ne, sw, se)
     
    #draws ellipses on the face for the left and right face
    def draw_ellipse1(self, (nw, ne, sw, se), image_cut):
        nwsw = (int(math.sqrt((nw[1]-sw[1])**2 + (nw[0]-sw[0])**2))/2, 
            int(abs(nw[0]-sw[0])/self.offset))
        nwswa = int(-1.0*math.degrees(math.tan(float(nw[0]-sw[0])/\
            (nw[1]-sw[1])))) + self.angle
        cv2.ellipse(image_cut, ((nw[0]+sw[0])/2, (nw[1]+sw[1])/2), \
            nwsw, nwswa, 0, self.angle*2, self.color, self.outline)
        nese = (int(math.sqrt((ne[1]-se[1])**2 + (ne[0]-se[0])**2))/2, \
            int(abs(ne[0]-se[0])/self.offset))
        nesea = int(-1.0*math.degrees(math.tan(float(ne[0]-se[0])/\
            (ne[1]-se[1])))) + (self.angle * 3)
        cv2.ellipse(image_cut, ((ne[0]+se[0])/2, (ne[1]+se[1])/2), \
            nese, nesea, 0, self.angle* 2, self.color, self.outline)
            
    #draws ellipses on the top and bottom of each cropped face
    def draw_ellipse2(self, (nw, ne, sw, se), image_cut):
        nwne = (int(math.sqrt((nw[0]-ne[0])**2 +(nw[1]-ne[1])**2))/2, 
            self.topoffset)
        nwnea = int(math.degrees(math.tan((nw[1]-ne[1])/float(nw[0]-ne[0])))) \
            + (self.angle * 2)
        cv2.ellipse(image_cut, ((nw[0]+ne[0])/2, (nw[1]+ne[1])/2), nwne,   
            nwnea, 0, self.angle * 2, self.color, self.outline)
        swse = (int(math.sqrt((sw[0]-se[0])**2 + (sw[1]-se[1])**2))/2, 
            self.bottomoffset)
        swsea = int(math.degrees(math.tan((sw[1]-se[1])/float(sw[0]-se[0]))))
        cv2.ellipse(image_cut, ((sw[0]+se[0])/2, (sw[1]+se[1])/2), swse, 
            swsea, 0, self.angle * 2, self.color,self.outline)    
            
    #controls the cropped shape for each image
    def draw_shape(self, (nw, ne, sw, se), image_cut):
        #details the color and outline width for each image
        self.color = (0, 255, 0, 255)
        self.outline = 2
        self.offset = 5.0
        self.angle = 90
        #calculates the top and bottom offset for rotated images
        self.topoffset = int(abs(nw[0]-ne[0])/self.offset)
        self.bottomoffset= int(abs(sw[0]-se[0])/self.offset)
        pts = np.array([nw, ne, se, sw], np.int32)
        pts = pts.reshape((-1,1,2))
        #draws a polygon around the face to bridge and gaps left by ellipses
        cv2.polylines(image_cut,[pts],True, self.color)
        self.draw_ellipse1((nw, ne, sw, se), image_cut)
        self.draw_ellipse2((nw, ne, sw, se), image_cut)
        return (self.topoffset, self.bottomoffset)
        
    #checks if the imputed color is within the color threshold
    #this is to identify lines drawn on each face
    def color_check(self, image_cut, x, y):
        color_thresh = self.color_thresh
        if color_thresh [0][0] <= image_cut[y][x][0] <= color_thresh [0][1] and\
            color_thresh [2][0] <= image_cut[y][x][2] <= color_thresh [2][1] \
            and color_thresh [1][0] <= image_cut[y][x][1] <=color_thresh [1][1]: 
            return True
        else:
            return False
        
    #uses edges to crop a clean version of an image    
    def cropedge(self, image_cut, image_cut_clean):
        #determines the best place to perform a tight image crop
        self.cutx = [[],[]]
        self.cuty = [[],[]]
        #determines the two directions for the checker to go
        for dir in [(0, len(image_cut[0]), 1, 0),
            (len(image_cut[0])-1, 0, -1, 1)]:
            #loops in vertical direction
            for y in xrange(len(image_cut)):
                alpha = True
                #loops through each pixel in horizontal direction
                for x in xrange(dir[0], dir[1], dir[2]):
                    #determines if an edge is found
                    if self.color_check(image_cut, x, y):
                        self.cuty[dir[3]].append(y)
                        self.cutx[dir[3]].append(x)
                        #creates bluring effect if the edge is found
                        for i in xrange(self.blur, -1, -1):
                            #finds rgb of the clean image
                            r = image_cut_clean[y][x+(i*dir[2])][0]
                            g = image_cut_clean[y][x+(i*dir[2])][1]
                            b = image_cut_clean[y][x+(i*dir[2])][2]
                            #increases the level of transparency for blurring 
                            #effect when a edge is found
                            image_cut_clean[y][x+(i*dir[2])] = \
                                np.array([r,g,b,i*self.blur*2])    
                        alpha = False
                        break
                    #if an edge hasn't been found, make the pixel transparent
                    if alpha == True:
                        image_cut_clean[y][x] = np.array([0,0,0,0])     
        return image_cut_clean
        
    #controller for cropping an image
    def crop_img(self, angle, image_cut):
        #opens a clean image adds transparency
        image_cut_clean = cv2.imread(self.result_path, -1)
        (row,col, n) = image_cut_clean.shape
        center=tuple(np.array([row,col])/2)
        rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
        #rotates the image appropriately to match previous images rotation
        image_cut_clean = cv2.warpAffine(image_cut_clean, rot_mat, (col,row))        
        image_cut_clean = self.cropedge(image_cut, image_cut_clean)
        #crops and saves the tightly cropped image
        cv2.imwrite(self.result_path,image_cut_clean[min(self.cuty[0]):\
        max(self.cuty[1]), min(self.cutx[0]):max(self.cutx[1])])    
     
    #finds the edges drawn on by cv2
    def detections_draw(self, face_pt, image_cut):
        (nw, ne, sw, se) = self.findpt(face_pt)
        red_thresh = (0, 110)
        green_thresh = (200, 255)
        blue_thresh = (0, 110)
        self.color_thresh = (red_thresh, green_thresh, blue_thresh)
        cv2.imwrite(self.result_path,image_cut)          
        (topoffset, bottomoffset) = self.draw_shape((nw,ne,sw,se), image_cut)
        (row,col, n) = image_cut.shape
        #rotates the image appropriately so the eyes are horizontal
        angle = int(math.degrees(math.tan(1.0*(face_pt[0][1]-face_pt[1][1])\
            /(face_pt[0][0] - face_pt[1][0]))))
        center=tuple(np.array([row,col])/2)
        rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
        image_cut = cv2.warpAffine(image_cut, rot_mat, (col,row))
        self.crop_img(angle, image_cut)
        return (nw, ne, se, sw, topoffset, bottomoffset)
        
#allows for text to be placed as an object        
class Text(object):   
        #modified from the 15-112 notes; citation: David Kosbie
    def textSize(self, text, font):
        # Note that tkFont.Font.measure(text) 
        #and tkFont.Font.metrics("linespace")
        # were both unreliable, producing wrong results in some cases.
        # This is a bit crufty, but generally works...
        root = Tk()
        canvas = Canvas(root, width=800, height=800)
        temp = canvas.create_text(0, 0, text=text, anchor=NW, font=font)
        (x0, y0, x1, y1) = canvas.bbox(temp)
        canvas.delete(temp)
        root.destroy()
        return (x1-x0, y1-y0)
    
    #initiates the text function by finding default values
    def __init__(self, x, y):
        self.text = "text"
        self.fontsize = 20
        self.font = "Arial"
        self.fontstyle = "bold"
        self.color = (0,0,0)
        box = self.textSize(self.text, font=(self.font, self.fontsize, \
            self.fontstyle))
        (self.x,self.y,self.w,self.h) = (x,y,box[0],box[1])
        self.edit = False
        
    #modified from 15-112 notes
    def rgbString(self, (red, green, blue)):
        return "#%02x%02x%02x" % (red, green, blue)        
        
    #represents the draw function for creating text   
    def draw(self, canvas):
        canvas.create_text(self.x+(self.w/2), self.y+(self.h/2), 
            text = self.text, font=(self.font, self.fontsize, self.fontstyle),
            fill=self.rgbString(self.color))
        #draws a box around text to show it is selected
        if self.edit == True:
            canvas.create_rectangle(self.x, self.y, self.x+self.w, 
                self.y+self.h, width=1, outline= "black")
    
    #color picker function for text color
    def colorpick(self):
        result = tkColorChooser.askcolor(color="#6A9662", 
                          title = "Pick a color") 
        self.color = result[0]       

    #saves the changes made via dialogue box
    def save(self):
        #sets fontsize, text value and resizes box
        self.fontsize = int(self.fontsizevar.get())
        self.text = self.textvar.get()        
        box = self.textSize(self.text, font=(self.font, self.fontsize, \
            self.fontstyle))
        (self.x,self.y,self.w,self.h) = (self.x,self.y,box[0],box[1])
            
    #removes editing box
    def exit(self):
        self.edit = False
        self.root.destroy()
            
    #creates dialogue box to edit text information
    def change(self):
        if self.edit == False:
            self.edit = True
            self.root = Tk()
            #dropdown to change size of font
            Label(self.root, text="Font Size: ").grid(row=0, column=0)            
            self.fontsizevar = StringVar(self.root)
            self.fontsizevar.set(str(self.fontsize))       
            menu = OptionMenu(self.root, self.fontsizevar, 
                "10","12","14","16","18","20","22","24","26","28","30",
                "32","34","36","38","40","42","44","46","48","50") 
            menu.grid(row=0, column=1, sticky=W)   
            #button to open color picker
            Button(self.root, text='Choose a Color',  fg="black", 
                   command=self.colorpick).grid(row=0, column=2)       
            Label(self.root, text="Text: ").grid(row=1, column=0)
            self.textvar = Entry(self.root)
            self.textvar.insert(0, self.text)
            self.textvar.grid(row=1, column=1)         
            #save and exit buttons to call their respective functions
            Button(self.root, text='Save', command=self.save, fg="Forest Green"
                ).grid(row=1, column=2)
            Button(self.root, text='Exit', command=self.exit, fg="red"
                ).grid(row=1, column=3)
            mainloop()
       
#class allowing users to create a paint object
class Paint(object):      
        
    #initiates the paint object with default values
    def __init__(self, x, y):
        self.color = "black"
        self.radius = 5
        self.points = []
        self.x = x
        self.y = y
        self.points.append((x,y))
        self.box = [x-self.radius, y-self.radius, x+self.radius, y+self.radius]
        self.w = self.radius
        self.h = self.radius
        self.edit = False
    
    #modified from 15-112 notes
    def rgbString(self, (red, green, blue)):
        return "#%02x%02x%02x" % (red, green, blue)       
    
    #determines size of the box around the painted picture
    def findbox(self):
        for (x,y) in self.points:
            if x < self.box[0]:
                self.box[0] = x
            if x > self.box[2]:
                self.box[2] = x
            if y < self.box[1]:
                self.box[1] = y
            if y > self.box[3]:
                self.box[3] = y
        self.x = self.box[0]
        self.y = self.box[1]
        self.w = self.box[2] - self.box[0]
        self.h = self.box[3] - self.box[1]
    
    #draws the represented paint on canvas
    def draw(self, canvas):
        for (x,y) in self.points:
            canvas.create_oval(x-self.radius,y-self.radius,x+self.radius,y+\
                self.radius, fill = self.color, outline=self.color)
        #creates box around paint on canvas to edit
        if self.edit == True:
            canvas.create_rectangle(self.box[0], self.box[1], self.box[2], \
                self.box[3], outline="black", width=1)
           
    #initializes color picker for the paint function           
    def colorpick(self):
        result = tkColorChooser.askcolor(color="#6A9662", 
                          title = "Pick a color") 
        self.color = self.rgbString(result[0])

    #allows the paint function to be saved
    def save(self):
        self.radius = int(self.radiusvar.get())        
        (self.x,self.y,self.w,self.h) = (self.x,self.y,self.box[0],self.box[1])
            
    #allows the dialogue box to be closed
    def exit(self):
        self.edit = False
        self.root.destroy()            
            
    #creates the dialogue box for the user to edit the painted image
    def change(self):
        self.edit = True
        self.root = Tk()
        #creates radius input
        Label(self.root, text="Radius: ").grid(row=0, column=0)            
        self.radiusvar = StringVar(self.root)
        self.radiusvar.set(str(self.radius))
        menu = OptionMenu(self.root, self.radiusvar, 
            "1","2","4","6","8","10","12","14","16","18","20") 
        menu.grid(row=0, column=1, sticky=W)   
        #allows user to pick a color
        Button(self.root, text='Choose a Color',  fg="black", 
               command=self.colorpick).grid(row=0, column=2)   
        #allows user to save and exit
        Button(self.root, text='Save', command=self.save, fg="Forest Green"
            ).grid(row=1, column=2)
        Button(self.root, text='Exit', command=self.exit, fg="red"
            ).grid(row=1, column=3)
        mainloop()            

#represents object for a face on the canvas
class DropFace(object):
    #initiates defaults for the image on the canvas
    def __init__(self, toolw, barh, x, y, w, h, f, faceimg, selected):
        self.edit = False
        self.x = toolw + f[0][0] + x
        self.y = f[0][1] + y + barh - f[4]
        #determines face width by distance between two top points of eyes
        self.w = int(math.sqrt((f[1][0] - f[0][0])**2 + (f[1][1] - f[0][1])**2))
        #calculates the vertical position front bottom and top offset
        self.h = f[4] + f[5] + f[2][1] - f[0][1]
        self.brightnessval = 1.00
        self.colorval = 1.00
        photo = PIL.Image.open(faceimg[selected][1])
        #resizes image based width and height found
        photo = photo.resize((self.w, self.h))
        #rotates image based on the delta angle between two eyes
        angle = int(-1.0*math.degrees(math.tan(float(f[0][1]-f[1][1])/\
            (f[0][0] - f[1][0]))))
        self.photo = photo.rotate(angle)
        self.img = ImageTk.PhotoImage(self.photo)  
        
    #allows for the edits to the image to be saved
    def save(self):
        #creates the scale for the change in brightness and coloration
        scale = 500
        self.brightnessval = self.brightnessval + \
            float(self.brightness.get())/scale
        #allows for user to change brightness and color saturation of a photo
        enhancer = ImageEnhance.Brightness(self.photo)
        self.photo = enhancer.enhance(self.brightnessval)  
        self.colorval = self.colorval + float(self.color.get())/scale
        enhancer = ImageEnhance.Color(self.photo)
        self.photo = enhancer.enhance(self.colorval)          
        self.img = ImageTk.PhotoImage(self.photo)  
        self.brightnessval = 1.00
        self.colorval = 1.00
        
    #allows for user to close dialogue box
    def exit(self):
        self.edit = False
        self.root.destroy()            
        
    #allows for user to change the brightness and color saturation of image
    def change(self):
        self.edit = True
        self.root = Tk()
        scale = 100
        #adds brightness slider
        Label(self.root, text="Brightness: ").grid(row=0, column=0)            
        self.brightness = Scale(self.root, from_=-scale, to=scale, 
                                orient=HORIZONTAL)
        self.brightness.set((self.brightnessval * scale)-scale)
        self.brightness.grid(row=0, column=1, sticky=W)
        #adds color saturation slider
        Label(self.root, text="Color: ").grid(row=1, column=0)            
        self.color = Scale(self.root, from_=-scale, to=scale, orient=HORIZONTAL)
        self.color.set((self.colorval * scale)-scale)
        self.color.grid(row=1, column=1, sticky=W)
        #save and exit buttons
        Button(self.root, text='Save', command=self.save, fg="Forest Green"
            ).grid(row=3, column=2)
        Button(self.root, text='Exit', command=self.exit, fg="red"
            ).grid(row=3, column=3)
        mainloop() 

    #draws the image on the canvas  
    def draw(self, canvas):
        canvas.create_image(self.x, self.y, anchor=NW, image=self.img)        
        
#creates the editor function        
class Editor(unsafeEventBasedAnimation.Animation):
    #initiates values to browse for an image
    def initBrowse(self):
        self.loaded = False
        self.loading = False    
        self.browseh = 20
        self.browsew = 200
        
    #initiates the picked faces selection
    def initPick(self):
        self.barh = 150        
        self.toolw = 100
        self.barmin = 0  
        self.pick = []  
        self.selected = None  
        self.pickoffset = 0
        self.arrowh = 40
        #determines where right and left arrows go
        self.arrow = [[0,self.barh-self.arrowh,self.arrowh,
            self.arrowh,self.toolw], [self.width-self.arrowh,
            self.barh-self.arrowh,self.arrowh,self.arrowh,-1*self.toolw]]
            
    #chooses where to put tools on the toolbar
    def initTools(self):
        button1 = 1.5
        button2 = 2.0
        button3 = 2.5
        button4 = 3.0
        button5 = 3.5
        self.export = [0, self.barh, self.toolw, self.barh + self.toolw/2]
        self.importf = [0, self.barh + self.toolw/2, 
            self.toolw, self.barh + self.toolw]
        self.showdropb = [0, self.barh + self.toolw, self.toolw, 
            self.barh + int(self.toolw * button1)]     
        self.erase = [0, self.barh + int(self.toolw*button1), self.toolw,
            self.barh + int(self.toolw * button2)]
        self.addtext = [0, self.barh + int(self.toolw*button2), self.toolw,
            self.barh + int(self.toolw * button3)]
        self.paint = [0, self.barh + int(self.toolw*button3), self.toolw,
            self.barh + int(self.toolw * button4)]
        self.save = [0, self.barh + int(self.toolw*button4), self.toolw,
            self.barh + int(self.toolw * button5)]            

    #controller for init function
    def onInit(self):
        self.cursor = (0,0)        
        self.step = 0  
        self.timerDelay = 50
        self.showdrop = False     
        self.showdropt = "Mark\nFaces"        
        self.centerx = self.width/2
        self.centery = self.height/2
        self.dropped = []   
        self.objdropped = []
        self.initBrowse()        
        self.initPick()
        self.initTools()
        self.action = None
        self.saveimg = [False, None]
        self.currdir = os.getcwd()

    #attempts to unzip a file with cropped images
    def unzip(self, import_path, imgs, zip, dir):
        with zipfile.ZipFile(import_path, 'r') as myzip:
            listfiles = myzip.namelist()
            #increases all filenames buy the number found
            for file in xrange(len(imgs)-1, -1, -1):
                num = os.path.splitext(imgs[file])[0]
                os.rename(zip + "/" + imgs[file], \
                    zip + "/" + str(file) + "_0" + ".png")
            #extracts files into folder, thus importing faces
            for file in xrange(len(listfiles)):
                myzip.extract(listfiles[file], dir + "/" + zip)    
        
    #imports files controller
    def importfile(self):
        dir = os.path.dirname(self.img_path)
        zip = os.path.splitext(os.path.basename(self.img_path))[0]
        imgs = os.listdir(self.faces.dir)
        currdir = os.getcwd()
        types = [('Zip Files', '.zip'), ('all files', '.*')]
        #creates dialogue to import files
        import_path = tkFileDialog.askopenfilename(initialdir=currdir, 
            title='Please select a zip to import', filetypes = types) 
        faces = os.listdir(dir + "/" + zip)
        #attempts to import the files from zip
        try:
            self.unzip(import_path, imgs, zip, dir)
        except:
            pass

    #creates a zip of current faces
    def makezip(self):
        dir = os.path.dirname(self.img_path)
        zip = os.path.splitext(os.path.basename(self.img_path))[0]
        #creates zip file and adds faces in
        with zipfile.ZipFile(zip + ".zip", 'w') as myzip:
            self.imgs = os.listdir(self.faces.dir)
            for img in self.imgs:
                myzip.write(dir + "/"  + zip + "/" + img, img)

    #scales the image based on the width of the window
    def scaleimg(self, image):
        #determines if image is larger than window
        if self.imwidth > self.width - self.toolw or self.imheight > \
            self.height - self.barh:
            #creates a scale factor
            scale = min(float(self.width - self.toolw)/self.imwidth, 
                        float(self.height - self.barh)/self.imheight)
            self.scaledwidth = int(self.imwidth*scale)
            self.scaledheight = int(self.imheight*scale)
            image = image.resize((self.scaledwidth, self.scaledheight))
        else:
            #otherwise, simply scales the image
            self.scaledwidth = self.imwidth
            self.scaledheight = self.imheight
        return image
                
    #allows user to browse for a file to edit
    def browse(self):
        currdir = os.getcwd()
        self.loading = True
        #creates dialogue to locate an image file
        self.img_path = tkFileDialog.askopenfilename(initialdir=currdir, \
            title='Please select an Image', filetypes = [('Image Files', \
                '*.jpg; *.gif; *.png'), ('all files', '.*')])
        base = os.path.basename(self.img_path)
        ext = os.path.splitext(base)[1]
        #checks if the file is a image file
        if len(self.img_path) > 0 and (ext is ".jpg" or ".gif" or ".png"):
            image = PIL.Image.open(self.img_path).convert('RGBA')
            (self.imwidth, self.imheight) = image.size
            image = self.scaleimg(image)
            #saves the file as a transparent image
            image.save(self.img_path, "PNG")
            self.loaded = True
            self.image = ImageTk.PhotoImage(file=self.img_path)
            #detects the faces in the image
            self.faces = Face(self.img_path, self.width, self.height)
            self.imgs = os.listdir(self.faces.dir)
            #opens each of the cropped faces
            self.faceimg = []
            for img in self.imgs:
                self.faceimg.append((ImageTk.PhotoImage(file=self.faces.dir \
                    + "/" + img), self.faces.dir + "/" + img))
        self.loading = False
            
    #saves the canvas as a postscript to be converted to a jpg
    def savefile(self):
        currdir = os.getcwd()
        savedir = tkFileDialog.asksaveasfilename(defaultextension=".eps")
        if savedir == None:
            return None
        self.saveimg = [True, savedir]
        
    #toggles the ability to mark users faces
    def toggledrop(self):
        if self.showdrop == True:
            self.showdrop = False
            self.showdropt = "Mark\nFaces"
        elif self.showdrop == False:
            self.showdrop = True
            self.showdropt = "Unmark\nFaces"
            
    #allows a user to erase faces on a image
    def clear(self, event):
        #checks for all dropped faces
        for i in xrange(len(self.dropped)-1,-1,-1):
            if self.dropped[i].x < event.x < self.dropped[i].x + \
                self.dropped[i].w and self.dropped[i].y < event.y \
                < self.dropped[i].y + self.dropped[i].h:
                self.dropped.remove(self.dropped[i])
        #checks for all dropped objects
        for i in xrange(len(self.objdropped)):
            if self.objdropped[i].x < event.x < self.objdropped[i].x + \
                self.objdropped[i].w and self.objdropped[i].y < event.y < \
                self.objdropped[i].y + self.objdropped[i].h:
                self.objdropped.remove(self.objdropped[i])
                
    #controls the ability for buttons on the toolbar to be clicked
    def mouseToolDrops(self, event):
        #controls eraser button
        if self.erase[0]< event.x < self.erase[2] and self.erase[1]\
            < event.y < self.erase[3]:
            if self.action != "eraseface":
                self.action = "eraseface"          
            elif self.action == "eraseface":
                self.action = None
        #controls text button
        if self.addtext[0]< event.x < self.addtext[2] and self.addtext[1]\
            < event.y < self.addtext[3]:
            if self.action != "textdrop":
                self.action = "textdrop"          
            elif self.action == "textdrop":
                self.action = None
        #controls paint button
        if self.paint[0]< event.x < self.paint[2] and self.paint[1]\
            < event.y < self.paint[3]:    
            if self.action != "initpaint":
                self.action = "initpaint"          
            elif self.action == "initpaint":
                self.action = None    
           
    #controller for the mousetoolbar           
    def mouseTool(self, event):
        #controls export button
        if self.export[0] < event.x < self.export[2] and self.export[1] \
            < event.y < self.export[3]:
            self.makezip()
        #controls import button
        if self.importf[0] < event.x < self.importf[2] and self.importf[1]\
            < event.y < self.importf[3]:
            self.importfile() 
        #controls the face circling button
        if self.showdropb[0]< event.x < self.showdropb[2] and self.showdropb[1]\
            < event.y < self.showdropb[3]:
            self.toggledrop()       
        #determines if the user wants to add an object to the canvas            
        self.mouseToolDrops(event)
        #controls saving face button
        if self.save[0]< event.x < self.save[2] and self.save[1]\
            < event.y < self.save[3]:    
            self.savefile()
             
    #on mouse click controller             
    def onMouse(self, event):
        #determines if the browse button was clicked
        if self.centerx - self.browsew < event.x < self.centerx + self.browsew \
            and self.centery - self.browseh < event.y < \
            self.centery + self.browseh and self.loaded == False:
            self.browse()
        if self.loaded == True:
            #controls the paint event
            if self.action == "initpaint":
                self.action = "paint"
                self.objdropped.append(Paint(event.x, event.y))        
            self.mouseTool(event)
            #controls the face drag and drop
            for (x,y,w,h,img,id) in self.pick:
                if x < event.x < x+w and y < event.y < y+h:
                    self.selected = id
            #controls picker navigation
            for (x,y,w,h,dir) in self.arrow:
                if x < event.x < x+w and y < event.y < y+h:
                    self.pickoffset += dir
            #controls the eraser button
            if self.action == "eraseface": self.clear(event)
        #determines if an face on the canvas is to be edited
        for obj in self.dropped:
            if obj.x < event.x < obj.x + obj.w and obj.y < event.y < obj.y \
                + obj.h and obj.edit == False: obj.change()                 
                    
    #checks for on mouse drag, if an item was dragged
    def onMouseDrag(self, event):
        #controls allowing text to be dropped on the canvas
        if self.selected != None or self.action == "textdrop":
            self.cursor = (event.x, event.y)    
        #controls the paint adding aspect
        if self.action == "paint":
            self.objdropped[-1].points.append((event.x, event.y))
        #checks if the object being moved is text
        for obj in self.objdropped:
            if repr(type(obj)) == "<class '__main__.Text'>" and obj.x < \
            event.x < obj.x + obj.w and obj.y < event.y < obj.y + obj.h:
                obj.x = event.x - (obj.w/2)
                obj.y = event.y - (obj.h/2)
    
    #transform face controller    
    def facetransform(self,x,y,w,h,f):
        #creates a new face and adds to to the dropped menu
        newface = DropFace(self.toolw, self.barh, 
            x, y, w, h, f, self.faceimg, self.selected)
        self.dropped.append(newface)
    
    #controller for mouse dropping
    def onMouseRelease(self, event):
        #checks if the face is dropped is legal
        if self.selected != None:
            for (x,y,w,h,f) in self.faces.fpos:
                if self.toolw + x < event.x < self.toolw + x + w and \
                    y +self.barh < event.y < y+h +self.barh:
                    self.facetransform(x,y,w,h,f)
            self.selected = None  
        #controls dropping text on the canvas
        if self.action == "textdrop":
            self.objdropped.append(Text(self.cursor[0], self.cursor[1]))
            self.action = None
        #controls painting on the canvas
        if self.action == "paint":
            self.objdropped[-1].findbox()
            self.action = None
        #controls dropping an item on the canvas if it should be edited
        for obj in self.objdropped:
            if obj.x < event.x < obj.x + obj.w and obj.y < event.y < obj.y \
                + obj.h and obj.edit == False:
                obj.change()                       

    #controls the stepper
    def onStep(self):
        self.step += 1
        #finds face images to load to the picker
        thumbnailsize = 100
        if self.loaded == True:
            facedir = self.currdir + "/" + self.faces.dir
            self.imgs = os.listdir(facedir)
            self.faceimg = []
            for img in self.imgs:
                self.faceimg.append((ImageTk.PhotoImage\
                (file=facedir + "/" + img), facedir + "/" + img))
            self.pick = []
            #controls the picker slider
            for id in xrange(self.barmin, len(self.faceimg)):
                photo = PIL.Image.open(self.faceimg[id][1])
                (w,h) = photo.size
                photo = photo.resize((min(thumbnailsize, w), 
                                      min(thumbnailsize, h)))
                image = ImageTk.PhotoImage(photo)
                self.pick.append((self.pickoffset + id*thumbnailsize, 
                                  0, w, h, image, id))    
 
    #allows for a canvas to be saved as a jpg 
    def saveimage(self, canvas):
        if self.saveimg[0] == True:
            #creates postscript file from canvas
            canvas.postscript(file=self.saveimg[1], colormode='color', 
                              pagewidth=self.width, pageheight=self.height)
            try:
                # attempts to open file
                im = PIL.Image.open(self.saveimg[1])
                path= os.path.basename(self.saveimg[1])
                filename = os.path.splitext(path)
                #crops out menubars from canvas
                im = im.crop((self.toolw, self.barh, self.scaledwidth + \
                    self.toolw, self.scaledheight + self.barh))
                #saves a file
                dir = os.path.dirname(self.saveimg[1])
                im.save(dir + "/" + filename[0] + ".jpg", "JPEG")
                self.saveimg = [False, None]
            except:
               self.saveimg = [False, None]
            
    #copied from notes    
    def rgbString(self, red, green, blue):
        return "#%02x%02x%02x" % (red, green, blue) 
            
    #based off example from notes
    def drawCircle(self, canvas, x, y, radius, color, outline_width):
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius, \
            fill=color, width=outline_width)            
           
    #loading screen, it is important to note that this is a modified version 
    #of the exercise in homework 3
    def drawLoading(self, canvas):
        background_grey = self.rgbString(200,200,200)
        radius = 150
        num_orbit = 8
        width = self.width
        height = self.height
        startcolor = 255
        numdots = 14
        dotradius = 25
        #computes the change in radians based on a step
        rad_delta = (math.radians(self.step)) + (3*math.pi/2)
        #creates center circle and text
        self.drawCircle(canvas, width/2, height/2, radius, background_grey, 0)
        canvas.create_text(width/2, height/2, text="Loading...", \
            font="Arial 22 bold")
        #creates the number of small orbiting circles 
        for amt in xrange(0, num_orbit):
            #computes the color based off the number of orbitiing circles
            color = (amt*(startcolor/(num_orbit - 1)))
            #computes the number of radians that the orbiting circles are at
            rads = rad_delta - (math.pi * amt * 2 / numdots)
            self.drawCircle(canvas, (width/2) + radius*math.cos(rads), 
                            (height/2) + radius*math.sin(rads), dotradius, 
                            self.rgbString(color, color, color), 2)        
        
    #draws the face picker bar
    def drawPicker(self, canvas):
        for (x,y,w,h,dir) in self.arrow:
            canvas.create_oval(x,y,x+w,y+h, fill="brown4")
            #draws the nav arrows
            if dir == self.toolw:
                txt = "<"
            elif dir == (-1*self.toolw):
                txt = ">"
            canvas.create_text(x+(w/2), y+(h/2), text=txt, \
                font="Arial 30 bold", fill="white")
                
    #draws button for the toolbar
    def drawFaceEditor(self, canvas):
        canvas.create_rectangle(self.export[0], self.export[1], self.export[2],
            self.export[3], fill="skyblue")
        canvas.create_text((self.export[0]+self.export[2])/2, 
            (self.export[1]+self.export[3])/2, text="Save\nFaces", 
            font = "Arial 16 bold")
        canvas.create_rectangle(self.importf[0], self.importf[1], 
            self.importf[2], self.importf[3], fill="magenta")
        canvas.create_text((self.importf[0]+self.importf[2])/2, 
            (self.importf[1]+self.importf[3])/2, text="Import\nFaces", 
            font = "Arial 16 bold")
        canvas.create_rectangle(self.showdropb[0], self.showdropb[1], 
            self.showdropb[2], self.showdropb[3], fill="DodgerBlue2")
        canvas.create_text((self.showdropb[0]+self.showdropb[2])/2, 
            (self.showdropb[1]+self.showdropb[3])/2, text=self.showdropt, 
            font = "Arial 16 bold") 
        canvas.create_rectangle(self.save[0], self.save[1], 
            self.save[2], self.save[3], fill="IndianRed1")
        canvas.create_text((self.save[0]+self.save[2])/2, 
            (self.save[1]+self.save[3])/2, text="Save", 
            font = "Arial 16 bold")             
            
    #draws the editor for each of the button in the toolbar
    def drawEditor(self, canvas):
        if self.action == "eraseface": color = "forest green"
        elif self.action != "eraseface": color = "red"
        canvas.create_rectangle(self.erase[0], self.erase[1], 
            self.erase[2], self.erase[3], fill=color)
        canvas.create_text((self.erase[0]+self.erase[2])/2, 
            (self.erase[1]+self.erase[3])/2, text="Clear", 
            font = "Arial 16 bold")
        canvas.create_rectangle(self.addtext[0], self.addtext[1], 
            self.addtext[2], self.addtext[3], fill="gold")
        canvas.create_text((self.addtext[0]+self.addtext[2])/2, 
            (self.addtext[1]+self.addtext[3])/2, text="Add\nText", 
            font = "Arial 16 bold")
        if self.action == "paint" or self.action == "initpaint":
            color = "forest green"
        elif self.action != "paint" or self.action != "initpaint":
            color = "red"            
        canvas.create_rectangle(self.paint[0], self.paint[1], 
            self.paint[2], self.paint[3], fill=color)
        canvas.create_text((self.paint[0]+self.paint[2])/2, 
            (self.paint[1]+self.paint[3])/2, text="Paint", 
            font = "Arial 16 bold")                 
        if self.action == "textdrop":
            canvas.create_text(self.cursor[0], self.cursor[1], text="text", 
                font="Arial 16 bold")
            
    #draws and is controller for drawing the toolbar
    def drawTools(self, canvas):
        canvas.create_rectangle(0, self.barh, self.toolw, self.height, 
            fill="orange")
        self.drawFaceEditor(canvas)
        self.drawEditor(canvas)
        
    #draws each object on the canvas
    def drawObjects(self, canvas):
        for object in self.objdropped:
            object.draw(canvas)
        
    #draws each of the faces on the canvas in picker or being moved
    def drawFaces(self, canvas):
        for (x,y,w,h,img,id) in self.pick:
            canvas.create_image(x,y, anchor=NW, image=img)
        if self.selected != None:
            canvas.create_image(self.cursor[0],self.cursor[1], anchor=CENTER, \
                image=self.faceimg[self.selected][0])
        for face in self.dropped:
            face.draw(canvas)
    
    #draws the splash screen for the program
    def drawBrowse(self, canvas):
        title="Quik Swap"
        fontsizetitle = 40
        fontsizesubtitle = 28
        titleratio = .67
        canvas.create_text(self.centerx, self.centery - (self.browsew), 
            text=title, font=("Comic Sans MS", fontsizetitle, "bold"), 
            fill="Sienna")
        subtitle="\"The Fast Face Swap\""
        canvas.create_text(self.centerx, self.centery - \
            int(self.browsew*titleratio), text=subtitle, 
            font=("Comic Sans MS", fontsizesubtitle, "bold"), fill="Sienna")
        canvas.create_rectangle(self.centerx - self.browsew, self.centery\
            - self.browseh, self.centerx + self.browsew, self.centery + \
            self.browseh, fill="forest green")
        msg = "Browse for an Image"
        canvas.create_text(self.centerx, self.centery, text=msg, \
            font="Arial 20 bold", fill="white")
            
    #draws circles and eyeline around each face
    def drawShadowDrop(self, canvas):
        outline = 2
        if self.showdrop == True:
            #draws circle and eyeline for each  face on canvas
            for (x,y,w,h,f) in self.faces.fpos:
                canvas.create_oval(self.toolw + x,y+self.barh,self.toolw + x+w\
                    ,y+h+self.barh,outline="lightgreen", width=outline)
                canvas.create_line((self.toolw + f[0][0]+x,f[0][1]+y+self.barh)\
                ,(self.toolw + f[1][0]+x, f[1][1] + y+self.barh), \
                width=outline, fill="red")
                
    #draw controller
    def drawPaint(self, canvas):
        canvas.create_image(self.toolw,self.barh, anchor=NW, image=self.image)
        self.drawFaces(canvas)
        self.drawPicker(canvas)
        self.drawTools(canvas)
        self.drawObjects(canvas)
        self.drawShadowDrop(canvas)
        self.saveimage(canvas)

    #head draw controller
    def onDraw(self, canvas):
        #checks if the image is loaded or not  
        if self.loading == True:
            self.drawLoading(canvas)
        else:
            if self.loaded == False:
                self.drawBrowse(canvas)
            else:
                self.drawPaint(canvas)

Editor(width=1300, height=800).run()