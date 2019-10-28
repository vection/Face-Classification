"""
   Face classification script
   Usage:
          python facedetector.py -l VideoPATH


   Made by Aviv Harazi
"""

import cv2
import face_recognition
import numpy as np
import ImageClass
import os
import argparse
import time


# General params
face_classifier = 'haarcascade_face.xml'
eye_classifier = 'haarcascade_eye.xml'
db = None
frame_steps = 1
frame_id = 0
frame = None


# Detection params
minimum_distance = 0.8
compare_distance = 0.68
resize_rate = 0.5
scale_factor = 1.2 # 1.2
min_neighbors = 3 # 3
min_size = (30,30) # (30,30)
eye_check = 1

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--link", action="store", dest="link", type=str,
                    help="provide video link")
parser.add_argument("-s", "--steps", dest="steps", type=int,
                    help="choose the frame steps (default = %d)" % frame_steps, default=frame_steps)
parser.add_argument("-r", "--resize", dest="resize", type=float,
                    help="choose resize rate (default = %f)" % resize_rate, default=resize_rate)
parser.add_argument("-e", "--eyes", dest="eye", type=int, choices=range(1,3),
                    help="Whether to clarify face with one or two eyes (default = %d)" % eye_check, default=eye_check)


# making sure all classifiers located in the right place
if not os.path.isfile('./%s' % eye_classifier) or not os.path.isfile('./%s' % face_classifier):
    exit("Haarcascade classifiers missing, please locate them with this script")

# installing classifiers
face_cascade = cv2.CascadeClassifier(face_classifier) # face classifier
eye_cacade = cv2.CascadeClassifier(eye_classifier) # eye classifier

"""
 if face is recognized by some limit its added to frames.txt of the registered image.
 if face is not recognized by the system as registered or the differences is too high, this face will be registered.
 
 :param list of faces represent as array
 :param current gray frame
 :param current rgb frame
"""
def check_faces(faces,frame,rgb):
    global db

    for (x, y, w, h) in faces:
            # encoding each face to the frame
            encoding = face_recognition.face_encodings(rgb, [(y, x + w, y + h, x)])[0]

            # comparing known faces to current face
            result = face_recognition.compare_faces(db.saved, encoding, tolerance=compare_distance)
            face_distances = face_recognition.face_distance(db.saved, encoding)

            # most closer result to recorded faces, if found it updates frame.txt and continue to next face.
            if len(face_distances) > 0:
                index = np.argmin(face_distances)  # minimum index
                min_value = min(face_distances)  # minimum distance
                if min_value < minimum_distance:
                    if result[index]:
                        coordinates = "%d,%d,%d,%d" % (x, y, x + w, y + h)
                        # update frames txt
                        db.write_frame(index, frame_id, coordinates)
                        continue

            # register new face
            if check_eye(frame[y:y + h, x:x + w]):
                db.insert(frame[y:y + h, x:x + w], encoding)

"""
    Checking if two eyes contains in frame, its second check to raise the chances we detect a face.
    
    :param frame
    :return True/False
"""
def check_eye(frame):
    global eye_check

    eyes = eye_cacade.detectMultiScale(frame)
    if len(eyes) > (eye_check-1):
        return True
    return False

# main function
def main(argp=None):
    global frame_id,db,frame_steps,resize_rate,eye_check

    if argp is None:
        argp = parser.parse_args()
    if argp.link is None:
        exit("Video link is need to be supplied")

    frame_steps = argp.steps
    resize_rate = argp.resize
    eye_check = argp.eye
    timer = time.time()
    video_path = argp.link

    # Read input video
    print("Face classification has been started on %s" %(video_path))
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        exit("There is a problem with video path, program terminated")

    # Creating db class
    db = ImageClass.Database(video_path)
    print("Processing")

    while True:
        for i in range(frame_steps):
            ret, frame = cap.read()
        if not ret:
            exit("Face classification finished after %d seconds, folder link %s\%s" % ((time.time() - timer), os.getcwd(), db.img_location))

        # resize each frame
        frame_resized = cv2.resize(frame, (0, 0), fx=resize_rate, fy=resize_rate)

        # gray/rgb frames
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # face detection using cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor,minNeighbors=min_neighbors, minSize=min_size)

        # checking if faces appear
        check_faces(faces,gray,rgb)
        frame_id += frame_steps

    cap.release()


if __name__ == '__main__':
    main()



