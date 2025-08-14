import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle

# Define a folder path where your training image dataset will be stored
path = "student_images"

# Create a list to store person_name and image array
# Traverse all image files present in the path directory, read images, 
# and append the image array to the image list and file name to classNames.
images = []
classNames = []

mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f"{path}/{cl}")
    if curImg is None:
        print(f"Could not open or find the image: {cl}")
        continue
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Create a function to encode all the train images and store them in a variable encoded_face_train
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Compute face encodings for all detected faces in the image
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            # Use the first detected face's encoding (assuming only one face per image)
            encodeList.append(face_encodings[0])
        else:
            # Handle cases where no face is detected
            print("FACE RECOGNITION ATTENDANCE SYSTEM")
    return encodeList

# Call the function to compute face encodings for your training images
encoded_face_train = findEncodings(images)

# Create a function that will create a Attendance.csv file to store the attendance with time
# Initialize an empty set to store recorded names
recorded_names = set()

def markAttendance(name):
    global recorded_names  # Use the global set

    if name not in recorded_names:
        # Record attendance
        with open('Attendance.csv', 'a') as f:
            now = datetime.now()
            time = now.strftime('%I:%M:%S %p')
            date = now.strftime('%d-%B-%Y')
            f.write(f'{name}, {time}, {date}\n')

        # Add the name to the set of recorded names
        recorded_names.add(name)


# take pictures from webcam 
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    
    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        
        if faceDist[matchIndex] < 0.6:  # Set a threshold for face distance
            name = classNames[matchIndex].upper().lower()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)  
    
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Change waitKey delay to 1 ms
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()