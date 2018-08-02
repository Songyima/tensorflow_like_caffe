#!/usr/bin/env python
# -*- coding: utf-8 -*-
from imutils.face_utils import FaceAligner
from dlib import rectangle
import face_recognition
import imutils
import dlib
import cv2
import os

root = 'msXGR/before_align'+'/'
root2 = 'msXGR/after_align'+'/'
 
def resize(img):
    resized = cv2.resize(img, (0,0), fx=2.5, fy=2.5)
    return resized
 
detector = dlib.get_frontal_face_detector()
#path to the dlib model shape_predictor_68_face_landmarks.dat
predictor = dlib.shape_predictor("align/shape_predictor_68_face_landmarks.dat")
# you can specify your photo_size after align using desiredFaceWidth
fa = FaceAligner(predictor, desiredFaceWidth=256)
all_ = 0 
folders = os.listdir(root)
for photo_folder in folders:
    photo_set = os.listdir(root+photo_folder)
    for im in photo_set:
        image_path = root+photo_folder+'/'+im
        print image_path
        all_+=1
        print 'photo nums now: ',all_
        img = cv2.imread(image_path)
        img = resize(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0)
     
        for j, face_location in enumerate(face_locations):
            # delete if j > 0 to multi faces
            if j > 0:
                break
            top, right, bottom, left = face_location
            dlib_rect = rectangle(left, top, right, bottom) #convert to dlib rect object
            faceOrig = imutils.resize(img[top:bottom, left:right], width=256)
            faceAligned = fa.align(img, gray, dlib_rect)        
            cv2.imwrite(root2+photo_folder+'/'+im,faceAligned)
