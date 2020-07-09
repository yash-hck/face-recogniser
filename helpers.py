import numpy as np
import glob
import os
from fr_utils import *
import cv2


def prepare_database(model):
    database = {}
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_to_encoding(file, model)

    return database


def add_to_database(imgname):
    name = imgname + '.jpg'
    path = os.path.join('images', name)

    video = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    PADDING = 25
    while True:
        _, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cord = face_cascade.detectMultiScale(gray, 1.5, 3)
        faces = []
        for x, y, w, h in face_cord:
            faces.append(frame[y-25:y+h+25, x-25:y+w+25])

        if len(faces) != 0:

            for x, y, w, h in face_cord:
                x1 = x - PADDING
                x2 = x+w + PADDING
                y1 = y - PADDING
                y2 = y+h+PADDING
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cut_image = frame[y1:y2, x1:x2]
            cv2.imwrite(path, cut_image)
            break

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    video.release()
    print('Ass')
    cv2.destroyAllWindows()


def recognise_face(imagepath, database, model):
    encoding = img_to_encoding(imagepath, model)
    identity = None
    min_dist = 100
    for (name, db_enc) in database.items():

        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' % (name, dist))
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.73:
        print('cant recognise the face', 2)
        return str(0)
    else:
        return str(identity)


'''def recognise_face(image, database, model):
    encoding = img_to_encoding(image, model)
    identity = 'unknown'
    min_dist = 100

    for (name, db_enc) in database.items():
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' % (name, dist))

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.6:
        print('can''t recognise face')
        return "unknown"
    else:
        return str(identity)
'''''