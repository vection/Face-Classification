import cv2
import random
import os
"""
    Database class meant to deal with the files
"""
class Database():
    def __init__(self, location):
        self.saved = []
        self.frames_path = []
        self.size = 0
        self.video_location = location
        self.img_location = "results_"+str(random.randint(1,1000))
        try:
            os.mkdir(self.img_location)
        except OSError:
            print("Creation of the directory failed")


    def insert(self, img, encoding_img):
        file_name = str(random.randint(0, 10000))
        path = self.openFolder(file_name)
        img_path = path+"/"+file_name+".jpg"
        frames_path = path+"/frames.txt"
        cv2.imwrite(img_path, img)
        ff = open(frames_path, "w+")
        self.size += 1
        self.saved.append(encoding_img)
        self.frames_path.append(frames_path)
        ff.close()

    def openFolder(self, name):
        try:
            path = self.img_location+"/"+str(name)
            os.mkdir(os.getcwd()+"\\"+self.img_location+"\\"+str(name))
            return path
        except OSError:
            print("Creation of the directory failed")


    def write_img(self, path,img):
        cv2.imwrite(path,img)

    def get(self, id):
        return self.saved[id]

    def write_frame(self, id, frame_number, coord):
        file = open(self.frames_path[id], "a")
        file.write("Frame: %d coordinates: %s \n" % (frame_number, coord))
        file.close()
