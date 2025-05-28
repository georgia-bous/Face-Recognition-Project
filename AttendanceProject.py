import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'ImagesBasic'
images = []
classNames = []

myList = os.listdir(path)  # List all files in the directory
print(myList)  # Print the list of files
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')  # Read each image
    images.append(curImg)  # Append the image to the list
    classNames.append(os.path.splitext(cl)[0])  # Append the class name (file name without extension)

print(classNames)  


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
        encode = face_recognition.face_encodings(img)[0]  
        encodeList.append(encode)  
    return encodeList  


def markAttendance(name):
    with open('attendance.csv', 'r+') as f: # read and write mode
        myDataList = f.readlines()  # Read all lines in the file
        nameList = []  
        for line in myDataList:
            entry = line.split(',')  
            nameList.append(entry[0])  
        if name not in nameList:  # If the name is not already in the list
            now = datetime.now()  
            dtString = now.strftime('%H:%M:%S')  # Format the time as a string
            f.writelines(f'\n{name},{dtString}')  # Write the name and time to the file



encodeListKnown = findEncodings(images)  
print("Encoding Complete with", len(encodeListKnown), "encodings") 


cap = cv2.VideoCapture(0)  # Start video capture from the webcam
while True:
    success, img = cap.read()  # Read a frame from the webcam
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize the image to speed up processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) 

    facesCurFrame = face_recognition.face_locations(imgS)  # Find faces in the current frame
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame) # If you do not pass the face locations to face_encodings, it will internally call face_locations again to detect faces
                                                                           # so i pass it  to speed up the process
    # Find encodings for the detected faces in the current frame            
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # Compare current face with known encodings
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  
        print(faceDis)  
        matchIndex = np.argmin(faceDis)  # Get the index of the closest match

        if matches[matchIndex]:  # If a match is found
            name = classNames[matchIndex].upper()  # Get the name of the matched person
            print(name)
            y1, x2, y2, x1 = faceLoc  
            y1 *= 4  #rescale to original size
            x2 *= 4  
            y2 *= 4  
            x1 *= 4  
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Draw rectangle around the face
            cv2.putText(img, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)  # Put name text on the image
            markAttendance(name)

    cv2.imshow("Webcam", img)  # Display the image with detected faces and names
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break