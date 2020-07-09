from inception_network import *
from fr_utils import *
import tensorflow as tf
from keras import backend as K
from helpers import *
import cv2
K.set_image_data_format('channels_first')


def triplet_loss_function(y_true, y_pred, alpha=0.3):
    anchor = y_pred[0]
    positive = y_pred[1]
    negative = y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


PADDING = 25
if __name__ == '__main__':
    model = model(input_shape=(3, 96, 96))
    model.compile(optimizer='adam', loss=triplet_loss_function, metrics=['accuracy'])

    load_weights_from_FaceNet(model)
    video = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    while True:
        _, frame = video.read()
        frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        key = cv2.waitKey(1)

        face_cord = face_cascade.detectMultiScale(frame_g, 1.5, 3)
        faces = []
        for (x, y, w, h) in face_cord:
            faces.append(frame[y - PADDING:y + h + PADDING, x - PADDING:x + w + PADDING])

        # if len(faces) == 0:
        #    cv2.imshow('frame', frame)
        #    if key == ord('q'):
        #        break
        #    else:
        #        continue
        if len(faces) != 0:
            for (x, y, w, h) in face_cord:
                x1 = x - PADDING
                y1 = y - PADDING
                x2 = x1 + w + PADDING
                y2 = y1 + h + PADDING

                cv2.rectangle(frame, (x - PADDING, y - PADDING), (x + w + PADDING, y + h + PADDING), (0, 255, 0), 2)
                cut_image = frame[y1:y2, x1:x2]
                cv2.imwrite('temp.jpg', cut_image)
                database = prepare_database(model)
                face = recognise_face('temp.jpg', database, model)

                cv2.putText(frame, face, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                os.remove("temp.jpg")

        cv2.imshow('frame', frame)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
