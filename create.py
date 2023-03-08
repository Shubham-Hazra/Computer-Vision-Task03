import cv2 as cv
import numpy as np

landmarks_path = '../data/captured/q3/landmarks.xml'

# Read the landmarks
landmarks = open(landmarks_path, 'r').read()
landmarks = landmarks.split('\n')
image_folder_path = '../data/masked/'
save_folder_path = '../data/captured/q3/synthesis/'
points = []
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]
for line in landmarks:
    line = line.strip()
    if line.startswith('<image file'):
        image_name = line.split('\'')[1].split('/')[-1]
        image_path = image_folder_path + image_name
        img = cv.imread(image_path)
        cv.imwrite(save_folder_path + image_name, img)
    else:
        if line.startswith('<part'):
            x = int(line.split('x=\'')[1].split('\'')[0])
            y = int(line.split('y=\'')[1].split('\'')[0])
            points.append((x, y))
        elif line.startswith('</image>'):
            points = np.array(points)
            for i, color in enumerate(colors):
                cv.polylines(img, [points], True, (0, 255, 0), 3)
                cv.fillPoly(img, [points],color)
                cv.imwrite(save_folder_path + image_name.split('.')[0] +"_"+ str(i)+'.jpg', img)
            points = []


