import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load Haar cascade files (keep XML files in same folder)
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")

st.title("👁️ Face & Eye Detection App (Haar Cascade)")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Convert to BGR (OpenCV format)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No Face Found 😢")
    else:
        st.success(f"{len(faces)} Face(s) Detected ✅")

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_classifier.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

    # Convert back to RGB for display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption="Detected Faces & Eyes", use_container_width=True)