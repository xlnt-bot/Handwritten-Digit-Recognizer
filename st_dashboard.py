import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from streamlit_drawable_canvas import st_canvas
import plotly.express as px

st.title("Handwritten Digit Recognizer")
st.markdown("""
Draw a digit (0-9) on the canvas below, and the AI will guess what it is!
""")

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('mnist_numbers.keras')
    except:
        return None
    
model = load_model()

if model is None:
    st.error("Error! Could not find 'mnist_numbers.keras' model. Please run training script first")
    st.stop()

st.sidebar.header("Canvas Settings")
stroke_width = st.sidebar.slider("Stroke Width",10,30,15)
stroke_color = "#ffffff"
bg_color = "#000000"
realtime_update = st.sidebar.checkbox("Update in realtime", True)

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Draw Here:")
    canvas_op = st_canvas(fill_color="rgba(255 ,165 ,0, 0.3)",
                       stroke_width=stroke_width,
                       stroke_color=stroke_color,
                       background_color=bg_color,
                       height=280,
                       width=280,
                       key = "canvas",
                       update_streamlit=realtime_update
                       )

#Creating a funtion to resize the images greter than 20 pixels and centering the ones outside detection box
def center_and_resize(image):
    cords = cv2.findNonZero(image)
    if cords is None:
        return image
    
    x,y,w,h = cv2.boundingRect(cords)
    digit = image[y:y+h, x:x+w]

    centered_image = np.zeros((28,28),dtype=np.uint8)

    if w > h:
        scale = 20.0/w
        dim = (20,int(h*scale))
    else:
        scale = 20.0/h
        dim = (int(w*scale),20)

    resized_digit = cv2.resize(digit, dim, interpolation=cv2.INTER_AREA)

    x_off = (28 - resized_digit.shape[1])//2
    y_off = (28 - resized_digit.shape[0])//2

    centered_image[y_off:y_off+resized_digit.shape[0], x_off:x_off+resized_digit.shape[1]] = resized_digit

    return centered_image

with col2:
    st.subheader("Prediction")

    if canvas_op.json_data is not None and len(canvas_op.json_data["objects"]) > 0:
            img_data = canvas_op.image_data

            grey_img = cv2.cvtColor(img_data.astype('uint8'),cv2.COLOR_RGBA2GRAY)

            final_img = center_and_resize(grey_img)

            if final_img.shape != (28,28):
                final_img = cv2.resize(final_img,(28,28))

            input_img = final_img/255.0  
            input_data = input_img.reshape(1,28,28,1)

            logits  = model.predict(input_data)
            probabilities = tf.nn.softmax(logits=logits).numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = np.max(probabilities)*100

            st.write(f"## **{predicted_class}**")
            st.write(f"Confidence: **{confidence:.2f}%**")
            
            chart_data = pd.DataFrame({
                'Digit': list(range(10)),
                'Probability': probabilities.tolist()
            })
            fig = px.bar(chart_data, x = 'Digit', y = 'Probability',
                         text_auto = '.1%',
                         height = 300)
            
            fig.update_traces(
                textposition='outside', # Put numbers on top of bars
                marker_color=['#E2E8F0' if i != predicted_class else '#3B82F6' for i in range(10)] # Highlight winner
            )
            
            # Clean up layout
            fig.update_layout(
                xaxis=dict(tickmode='linear', tick0=0, dtick=1, title="Digit"),
                yaxis=dict(range=[0, 1.1], showgrid=False, title="Confidence"),
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )

            st.plotly_chart(fig, width='stretch')
            
            #Showing what the model sees 
            with st.expander("See what the AI sees"):
                st.image(final_img, caption="Processed Image (28x28)",width = 'stretch')

    else:
        st.info("Draw something on the canvas to get started!")
    

