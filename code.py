import cv2
#for the camera
import numpy as np
#for maths 
import pandas as pd
#make dataframe
import seaborn as sns
#to make the graphs attractive
import matplotlib.pyplot as plt
#to plot the graph
from sklearn.datasets import fetch_openml
#to fetch the data from the open ml repo
from sklearn.model_selection import train_test_split
#split the data into train and test sets
from sklearn.linear_model import LogisticRegression
#creating log reg. model or classifier
from sklearn.metrics import accuracy_score
#to measure the accuracy of the model that we create

from PIL import Image
import PIL.ImageOps

import os, ssl, time

if (not os.environ.get('PYTHONHTTPSVERIFY', '')and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
#print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
n_classes = len(classes)

samples = 5
#how many samples we need from each label

figure = plt.figure(figsize = (n_classes*2, (1+samples*2)))
idx = 0
for cls in classes:
  #the labels in the classses array (0,1,2)
  indexes = np.flatnonzero(y==cls)
  #get a flattened (1D) array of indexes of all the images where label is 'cls'
  indexes = np.random.choice(indexes, samples, replace=False)
  #selecting 5 samples from all the indexes
  i = 0
  for index in indexes:
    plotindex = i*n_classes + idx + 1
    #where to place the same label/number
    #need to place 0 at position 1, then next 0 as position 11
    #the position in the graph where we want to place the current image.
    p = plt.subplot(samples, n_classes, plotindex)
    p = sns.heatmap(np.reshape(X[index], (22,30)), xticklabels=False, yticklabels=False, cmap=plt.cm.gray)
    p = plt.axis('off')
    i = i + 1
  idx = idx + 1


x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=7500, test_size = 2500, random_state = 9)
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0
#gives all the values between 0,1

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train_scaled, y_train)
#saga works best with multinomial

predictions = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)


cap = cv2.VideoCapture(0)
while(True):
    try:
        ret, frame = cap.read()
        #ret has true or false( read properly or not)
        #frame has the frame value

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #convert all the colours to grayscale

        height, width = gray.shape
        upperLeft = int(width/2 - 56), int(height/2 - 56)
        bottomRight = int(width/2 + 56), int(height/2 + 56)

        cv2.rectangle(gray, upperLeft, bottomRight, (0,255,0), 2)
        roi = gray[upperLeft[1]:bottomRight[1],upperLeft[0]:bottomRight[0]]

        #Converting cv2 image to pil format
        im_pil = Image.fromarray(roi)

        # convert to grayscale image - 'L' format means each pixel is 
        # represented by a single value from 0 to 255
        image_bw = im_pil.convert('L')
        image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
        max_pixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = clf.predict(test_sample)
        print("Predicted class is: ", test_pred)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
    
