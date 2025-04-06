import cv2
import face_recognition as fr
import numpy as np
import os
from datetime import datetime

# Folder containing known face images
path = 'resources'
images = []
classNames = []

# Only process valid image files
valid_extensions = ['.jpg', '.jpeg', '.png']
myList = [f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in valid_extensions]

for cl in myList:
    full_path = os.path.join(path, cl)
    curImg = cv2.imread(full_path)
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    else:
        print(f"[WARNING] Could not load image: {full_path}")

def findEncodings(images):
    encodeList = []
    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = fr.face_encodings(img)
        if faces:
            encodeList.append(faces[0])
        else:
            print(f"[WARNING] No face found in image: {classNames[i]}")
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding complete")

def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        f.seek(0)
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'{name},{dtString}\n')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame from webcam.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = fr.face_locations(imgS)
    encodesCurFrame = fr.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = fr.compare_faces(encodeListKnown, encodeFace)
        faceDis = fr.face_distance(encodeListKnown, encodeFace)

        if faceDis.size == 0:
            continue

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

            markAttendance(name)

    cv2.imshow("Webcam", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
