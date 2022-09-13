import streamlit as st
import os
import cv2
from PIL import Image

from pose_estimation import PoseEstimation
from yolo import *


os.chdir(os.getcwd())

filename = "input.jpg"

#value = st.slider('val')  # this is a widget
#st.write(value, 'squared is', value * value)

def save_uploadedfile(uploadedfile):
    global filename
    #filename = uploadedfile.name

    with open(os.path.join("temp","input.jpg"),"wb") as f:
        f.write(uploadedfile.getbuffer())    
    

    return st.success("Successfuly uploaded file ")
     

image_file = st.file_uploader("Enter a full-shot image", type = ['png','jpeg','jpg'])

if image_file is not None :
    save_uploadedfile(image_file)

    ps = PoseEstimation()
    print(filename)

    image_path = os.path.join("temp", filename)
    print(image_path)    

    if ps.driver(image_path) :
        st.write("Full shot")
        driver('temp/input.jpg')
        yolo_output = Image.open('temp/yolo_output.jpg')
        st.image(yolo_output, caption='Input Image')
    else:
        st.error("Not  full shot")

    #wd = YoloDetection(cv2.imread(os.path.join("temp", filename)))
    #wd.control_loop()
    
    
    #st.image(image_file, "Input Image")








