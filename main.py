#Main file for face swap
#created by Jeremy Lee (jalee)

import sys
import cv2
import numpy as np

def cascade_detect(cascade, image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(
        gray_image,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (10, 10),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
 
def detections_pt(image, path, detections, cascade_eye, cascade_mouth):
    for r in xrange(len(detections)):
        face_pt = []
        found = detections[r]
        (x,y,w,h) = (found[0], found[1], found[2], found[3])
        #face_pt.append((x,y,w,h))
        result_path = path[:-4] + str(r) + "_edit.png"
        image_cut = image[y:y+h, x:x+w]
        detections_eye = cascade_detect(cascade_eye, image_cut)
        if len(detections_eye) >= 0:
            for eye in xrange(2):
                (xe,ye,we,he) = (detections_eye[eye][0], detections_eye[eye][1], detections_eye[eye][2], detections_eye[eye][3])
                face_pt.append((xe,ye,we,he))
            detections_mouth = cascade_detect(cascade_mouth, image_cut)
            if len(detections_mouth) > 0:
                pot_mouths = []
                for mouth in xrange(len(detections_mouth)):
                    pot_mouths.append(detections_mouth[mouth][1])
                i = pot_mouths.index(max(pot_mouths))
                (xm,ym,wm,hm) = (detections_mouth[i][0], detections_mouth[i][1], detections_mouth[i][2], detections_mouth[i][3])
                face_pt.append((xm,ym,wm,hm))
                detections_draw(face_pt, image_cut, result_path)
 
def orderpt(face_pt):
    if face_pt[0][0] > face_pt[1][0]:
        return [face_pt[1]] + [face_pt[0]] + [face_pt[2]]
    return face_pt
 
def detections_draw(face_pt, image, result_path):
    face_pt = orderpt(face_pt)
    print face_pt
    for (x,y,w,h) in face_pt:  
        cv2.circle(image, (x+(w/2), y+(h/2)), w/2, (0, 255, 0), 2)     
    #cv2.line(image, ((face_pt[0][0])/2, face_pt[0][1]), ((face_pt[2][0]), face_pt[2][1] + (face_pt[2][2]/2)), (0, 255, 0), 2)
    #cv2.line(image, ((face_pt[0][0]), face_pt[0][1]), ((face_pt[1][0]+(face_pt[1][1])), face_pt[1][1]), (0, 255, 0), 2)
    #cv2.line(image, ((face_pt[1][0]+face_pt[1][2]), face_pt[1][1]), ((face_pt[2][0]+(face_pt[2][1]/2)), face_pt[2][1] + (face_pt[2][2]/2)), (0, 255, 0), 2)
    #cv2.polylines(image, [((face_pt[2][0]), face_pt[2][1] + (face_pt[2][2]/2)), ((face_pt[0][0])/2, face_pt[0][1]), ((face_pt[1][0]+(face_pt[1][1])), face_pt[1][1]), ((face_pt[2][0]+(face_pt[2][1]/2)), face_pt[2][1] + (face_pt[2][2]/2))], (0, 255, 0), 2)
    pts = np.array([((face_pt[2][0]), face_pt[2][1] + (face_pt[2][2]/2)), ((face_pt[0][0])/2, face_pt[0][1]), ((face_pt[1][0]+(face_pt[1][1])), face_pt[1][1]), ((face_pt[2][0]+(face_pt[2][1]/2)), face_pt[2][1] + (face_pt[2][2]/2))], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(image,[pts],True,(0, 0,255))
    cv2.imwrite(result_path, image)
 
def main():
    image_path = "Untitled.png"
    cascade_face_path = "haarcascade_frontalface_default.xml"
    cascade_eye_path = "haarcascade_eye.xml"
    cascade_mouth_path = "haarcascade_mouth.xml"
    result_path = "Untitled_edit.png"
    cascade_face = cv2.CascadeClassifier(cascade_face_path)
    cascade_eye = cv2.CascadeClassifier(cascade_eye_path)
    cascade_mouth = cv2.CascadeClassifier(cascade_mouth_path)
    image = cv2.imread(image_path)
    detections_face = cascade_detect(cascade_face, image)
    detections_pt(image, image_path, detections_face, cascade_eye, cascade_mouth)
    print "Found {0} objects!".format(len(detections_face))
    cv2.imwrite(result_path, image)
 
if __name__ == "__main__":
    sys.exit(main())