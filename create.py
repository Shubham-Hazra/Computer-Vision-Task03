import os

import cv2 as cv
import numpy as np

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]

def synthesize_image(image_path):
    image_name = image_path.split("/")[-1].split(".")[0]
    landmarks = []
    os.system("../dlib-master/examples/build/detect-shape shape_predictor.dat " + image_path)
    f = open("predicted.txt", "r")
    for line in f:
        line = line.strip()
        if line.startswith("face"):
            landmarks.append([])
        else:
            line = line.split(",")
            x = int(line[0][1:])//2
            y = int(line[1][1:-1])//2
            landmarks[-1].append((x, y))
    f.close()
    image = cv.imread(image_path)
    for i,color in enumerate(colors):
        if len(landmarks) == 0:
            break
        for landmark in landmarks:
            if len(landmark) == 0:
                continue
            cv.polylines(image, [np.array(landmark)], True, (0, 255, 0), 2)
            cv.fillPoly(image, [np.array(landmark)], color)
        cv.imwrite("../data/captured/q3/synthesis/" + image_name + "_synthesized"+str(i)+".jpg", image)

print("Do you want to create from landmarks or predict using shape predictor?")
print("1. Landmarks")
print("2. Shape Predictor")
choice = input("Enter your choice: ")
if choice == "1":
    landmarks_path = '../data/captured/q3/landmarks.xml'
    modified_landmarks_path = '/mnt/c/Users/shubh/onedrive/desktop/cs763/Task_03/data/captured/q3/landmarks_modified.xml'
    f = open(modified_landmarks_path, 'w')
    landmarks = open(landmarks_path, 'r').read()
    landmarks = landmarks.split('\n')
    image_folder_path = '../data/masked/'
    save_folder_path = '/mnt/c/Users/shubh/onedrive/desktop/cs763/Task_03/data/captured/q3/synthesis/'
    points = []
    landmark_lines = []
    save_line = False
    for line in landmarks:
        line_mod = line.strip()
        if line_mod.startswith('<image file'):
            save_line = True
        if save_line:
            landmark_lines.append(line)
        else:
            f.write(line)
            f.write('\n')
        if line_mod.startswith('<image file'):
            save_line = True
            image_name = line.split('\'')[1].split('/')[-1]
            image_path = image_folder_path + image_name
            img = cv.imread(image_path)
            cv.imwrite(save_folder_path + image_name, img)
            points = []
        else:
            if line_mod.startswith('<part'):
                x = int(line.split('x=\'')[1].split('\'')[0])
                y = int(line.split('y=\'')[1].split('\'')[0])
                points.append((x, y))
            elif line_mod.startswith('</image>'):
                for j,line_ in enumerate(landmark_lines):
                    f.write(line_)
                    f.write('\n')
                if len(points) == 0:
                    landmark_lines = []
                    continue
                points = np.array(points)
                for i, color in enumerate(colors):
                    for j,line_ in enumerate(landmark_lines):
                        line_stripped = line_.strip()
                        if line_stripped.startswith('<image file'):
                            line_ = line_.split('\'')[0] + '\'' + save_folder_path + image_name.split('.')[0] +"_"+ str(i)+'.jpg' + '\'' +('\'').join(line_.split('\'')[2:]) 
                            f.write(line_)
                            f.write('\n')
                        else:
                            f.write(line_)
                            f.write('\n')
                    cv.polylines(img, [points], True, (0, 255, 0), 3)
                    cv.fillPoly(img, [points],color)
                    cv.imwrite(save_folder_path + image_name.split('.')[0] +"_"+ str(i)+'.jpg', img)
                points = []
                landmark_lines = []
    f.write('</images>')
    f.write('</dataset>')
    f.close()

    print("Do you want to modify the xml file? (y/n)")
    ans = input()
    if ans == 'y':
        os.remove(landmarks_path)
        os.rename(modified_landmarks_path, landmarks_path)
    else:
        pass
elif choice == "2":
    print("Do you want to predict over an image or a folder?")
    print("1. Image")
    print("2. Folder")
    choice = input("Enter your choice: ")
    if choice == "1":
        print("Enter the path of the image")
        path = input("Enter the path: ")
        image_path = path
        synthesize_image(image_path)
    elif choice == "2":
        print("Enter the path of the folder")
        path = input("Enter the path: ")
        folder_path = path
        for image_path in os.listdir(folder_path):
            if image_path.endswith(".jpg"):
                image_path = folder_path + "/" + image_path
                synthesize_image(image_path)
            else:
                continue



