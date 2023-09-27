import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from keras_preprocessing import image
from keras.models import load_model
# from keras.applications.vgg19 import preprocess_input
# from tf.keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

st.title("𝐏𝐍𝐄𝐔𝐌𝐎𝐍𝐈𝐀 𝐏𝐑𝐄𝐃𝐈𝐂𝐓𝐈𝐎𝐍")

model = tf.keras.models.load_model("C:/Users/dj1058/OneDrive - CommScope/Documents/S7 Project/pneumonia_detection_model_1.h5")
### load file
uploaded_file = st.file_uploader("𝐂𝐡𝐨𝐨𝐬𝐞 𝐚 𝐢𝐦𝐚𝐠𝐞 𝐟𝐢𝐥𝐞", type=("jpg","jpeg","png"))

map_dict = {0: 'Normal',
            1: 'Pneumonia'}


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("𝐂𝐇𝐄𝐂𝐊 𝐑𝐄𝐒𝐔𝐋𝐓")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.success("𝐏𝐫𝐞𝐝𝐢𝐜𝐭𝐞𝐝 𝐑𝐞𝐬𝐮𝐥𝐭 𝐟𝐨𝐫 𝐭𝐡𝐞 𝐢𝐦𝐚𝐠𝐞 𝐢𝐬 {}".format(map_dict [prediction]))