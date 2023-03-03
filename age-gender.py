import argparse
import os
import sys

import cv2 as cv
import numpy as np

arguments = argparse.ArgumentParser()
arguments.add_argument('-i', help='Path to input image', required=True)

args = arguments.parse_args()

image_path = args.i
image = cv.imread(image_path)
(height, width) = image.shape[:2]

mean_values=(104, 117, 123)

faceNet = cv.dnn.readNetFromCaffe('deploy.prototxt','face.caffemodel')
ageNet = cv.dnn.readNetFromCaffe('age.prototxt','age.caffemodel')
genderNet = cv.dnn.readNetFromCaffe('gender.prototxt','gender.caffemodel')

face_blob = cv.dnn.blobFromImage(cv.resize(image,(300,300)), scalefactor=1.0, size=(300, 300),mean = mean_values, swapRB=False, crop=False)
faceNet.setInput(face_blob)
face_detections = faceNet.forward()
threshold = 0.8
for i in range(0, face_detections.shape[2]):
    confidence = face_detections[0, 0, i, 2]
    if confidence > threshold:
        box = face_detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        (startX, startY, endX, endY) = box.astype("int")
        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]
        if fW < 20 or fH < 20:
            continue
        blob = cv.dnn.blobFromImage(cv.resize(face,(224,224)), scalefactor=1.0, size=(224, 224),
                                                    mean=mean_values, swapRB=False, crop=False)
        ageNet.setInput(blob)
        genderNet.setInput(blob)
        age_detections = ageNet.forward()
        gender_detections = genderNet.forward()
        for ages in age_detections:
            predicted_age = np.argmax(ages)
            if predicted_age < 13:
                age = 'CHILD'
            elif predicted_age <= 19:
                age = 'TEEN'
            elif predicted_age <= 28:
                age = 'IITIAN'
            elif predicted_age <= 40:
                age = 'MIDDLE'
            elif predicted_age <= 55:
                age = 'BOOMER'
            elif predicted_age <= 65:
                age = 'DOWNHILL'
            elif predicted_age <= 75:
                age = 'SENIOR'
            else:
                age = 'SUPER'
        for genders in gender_detections:
            gender = np.argmax(genders)
            if gender == 0:
                sex = "F"
            else:
                sex = "M"
        cv.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)
        cv.putText(image,age+" "+sex, (startX + 10, endY - 10), cv.FONT_HERSHEY_SIMPLEX,1.2, (0, 255, 0), 2)
image_name = image_path.split("/")[-1]
output_path = "../results/age-gender/" + image_name
folders = os.listdir("../results")
if "age-gender" in folders:
    pass
else:
    os.mkdir("../results/age-gender")
cv.imwrite(output_path,image)



    

