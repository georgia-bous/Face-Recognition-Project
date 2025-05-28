import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file("ImagesBasic/ElonMusk.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("ImagesBasic/Elon test.jpg") # modify the test image
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
imgElonYoung = face_recognition.load_image_file("ImagesBasic/Elon young.jpg")
imgElonYoung = cv2.cvtColor(imgElonYoung, cv2.COLOR_BGR2RGB)
imgElonTeen = face_recognition.load_image_file("ImagesBasic/Elon teen.jpg")
imgElonTeen = cv2.cvtColor(imgElonTeen, cv2.COLOR_BGR2RGB)

# imgFriends = face_recognition.load_image_file("ImagesBasic/Friends.jpg")
# imgFriends = cv2.cvtColor(imgFriends, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]  # Get the first face location, (top, right, bottom, left) coordinates
                                                        # define the rectangle around the face in the image, in (row, column) space.
#print(faceLoc) 
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)  # Draw rectangle around the face, color, thickness

encodeElon = face_recognition.face_encodings(imgElon)[0]  # Get the first face encoding, its a 128 embedding
encodeTest = face_recognition.face_encodings(imgTest)[0]
encodeElonYoung = face_recognition.face_encodings(imgElonYoung)[0]
encodeElonTeen = face_recognition.face_encodings(imgElonTeen)[0]
results1 = face_recognition.compare_faces([encodeElon, encodeElonYoung, encodeElonTeen], encodeTest)  # Compare the encodings, returns a list of boolean values
print(results1)  # Print the results, True if the faces match, False otherwise

faceDis1 = face_recognition.face_distance([encodeElon, encodeElonYoung, encodeElonTeen], encodeTest)  # Get the face distance, lower value means more similar
print(faceDis1)  

results2 = face_recognition.compare_faces([encodeElonYoung], encodeElonTeen)  
print(results2)  
faceDis2 = face_recognition.face_distance([encodeElonYoung], encodeElonTeen)
print(faceDis2)

cv2.putText(imgTest, f'{results1}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)  # Put text on the image, text, position, font, scale, color, thickness

cv2.imshow("Elon Musk", imgElon) # display the image
cv2.imshow("Elon Test", imgTest)  # display the test image
cv2.waitKey(0)  # cv2.waitKey(0) pauses the program until you press any key, cv2.waitKey(1000)-> waits for 1000 ms 
                # Without waitKey, the image window would appear and disappear instantly
