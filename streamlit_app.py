import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import mediapipe as mp
import cv2
from keras.optimizers import Adam
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

model1 = tf.keras.models.load_model(r"E:/Proshekt/fully_trained_multi_v3.h5", compile=False)
model1.compile(optimizer = Adam(learning_rate=0.0001), loss = 'categorical_crossentropy', metrics=['accuracy'])

def load_one_image(path):
  img = Image.open(path)
  img = img.resize((224,224))
  img = np.array(img)
  return img

def face_crop(img, res):
    ans = []
    if res.detections:
        for detection in res.detections:
            bounding_box = detection.location_data.relative_bounding_box
            height, width, _ = img.shape
            xmin = int(bounding_box.xmin * width)
            ymin = int(bounding_box.ymin * height)
            w = int(bounding_box.width * width)
            h = int(bounding_box.height * height)
            xmax = xmin + w
            ymax = ymin + h
            face = img[ymin:ymax, xmin:xmax]
            ans.append(Image.fromarray(face))
    if ans != []:
        return ans
    img = Image.fromarray(img)
    return [img]

labels = ["FAKE", "REAL"]

def predictor(img, res):
    ofaces = face_crop(img, res)
    ofaces = [np.array(face.resize((224,224))) for face in ofaces]
    faces = [preprocess_input(face) for face in ofaces]
    st.markdown(f"### Faces found: **:green[{len(faces)}]**")
    results = model1.predict(np.array(faces))
    print(results)
    for ind, face in enumerate(ofaces):
        # print(results[ind])
        # print(type(results[ind]))
        # print(results[ind].shape)
        st.image(face)
        prediction = labels[np.argmax(results[ind])]
        st.markdown(f"## Prediction: **:blue[{prediction}]**")
        st.write(pd.DataFrame(results[ind].T,columns=["Probability"],index=labels))
        


# Upload the image file


uploaded_file = st.file_uploader("Choose an image file with a visible face")

min_conf = 0.5


# Finds face only
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=min_conf)
mp_drawing = mp.solutions.drawing_utils









# Creates mesh also
# holistic = mp.solutions.holistic
# holis = holistic.Holistic()
# drawing = mp.solutions.drawing_utils

if uploaded_file is not None:

    # Open the image using Pillow

    image = Image.open(uploaded_file).convert('RGB')
    st.write("Original Shape:", image.size)
    full_res_img = np.array(image)


    # res2 = holis.process(full_res_img)
    # if res2.face_landmarks:
    #     print(type(res2.face_landmarks))
    #     drawing.draw_landmarks(full_res_img, res2.face_landmarks)
    #     st.image(full_res_img)


    res = face_detection.process(full_res_img)

    if res.detections and len(res.detections) > 0:
        st.image(full_res_img, "Face Exists")
        predictor(full_res_img, res)
    else:
        st.image(full_res_img)
        if not res.detections:
            lenny = "NONE"
        else:
            lenny = len(res.detections)
        st.markdown(f"# :red[Wrong Image!]\n## There needs atleast :red[ONE] face in the image but you have given **:blue[{lenny}]**")
