import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Brain Tumor MRI Detection",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 90% 20%,
            rgba(0, 120, 255, 0.22),
            transparent 28%
        ),
        radial-gradient(
            circle at 10% 85%,
            rgba(120, 40, 255, 0.20),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #020817,
            #061b3d,
            #020817
        );
}

/* Main width */
.block-container {
    max-width: 1100px;
    padding-top: 35px;
}

/* Title */
.title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

.title-blue {
    background: linear-gradient(
        90deg,
        #18a8ff,
        #7557ff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #a9bad8;
    font-size: 19px;
    margin-bottom: 35px;
}

/* Main card */
.card {
    background: rgba(7, 25, 55, 0.75);
    border: 1px solid rgba(70, 150, 255, 0.35);
    border-radius: 24px;
    padding: 30px;
    box-shadow: 0 0 50px rgba(0, 100, 255, 0.15);
}

/* Upload heading */
.upload-heading {
    color: white;
    font-size: 23px;
    font-weight: 700;
}

.upload-text {
    color: #91a8cc;
    font-size: 14px;
    margin-bottom: 15px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(3, 17, 40, 0.8);
    border: 1px dashed rgba(80, 165, 255, 0.6);
    border-radius: 16px;
    padding: 10px;
}

/* Predict button */
.stButton > button {
    width: 100%;
    height: 60px;

    border: none;
    border-radius: 15px;

    background: linear-gradient(
        90deg,
        #1479ff,
        #7045ff
    );

    color: white;
    font-size: 20px;
    font-weight: 700;

    box-shadow: 0 8px 25px rgba(45, 90, 255, 0.35);
}

.stButton > button:hover {
    box-shadow: 0 12px 35px rgba(80, 100, 255, 0.5);
}

/* Image */
[data-testid="stImage"] {
    border-radius: 18px;
    overflow: hidden;
}

/* Result area */
.result-heading {
    text-align: center;
    color: #a9bad8;
    font-size: 18px;
    margin-top: 25px;
}

.footer {
    text-align: center;
    color: #61799f;
    font-size: 13px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "brain_tumor_cnn.h5"
    )

model = load_model()

# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image):

    # RGB
    image = image.convert("RGB")

    # Same size as training
    image = image.resize((224, 224))

    # NumPy
    image_array = np.array(image)

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Predict
    prediction = model.predict(
        image_array,
        verbose=0
    )

    # Class index
    predicted_index = np.argmax(
        prediction[0]
    )

    # Class name
    predicted_class = class_names[
        predicted_index
    ]

    # Confidence
    confidence = (
        prediction[0][predicted_index] * 100
    )

    return predicted_class, confidence

# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="title">
        🧠 Brain Tumor
        <span class="title-blue">
            MRI Detection
        </span>
    </div>

    <div class="subtitle">
        AI-powered MRI analysis for automated brain tumor classification.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# MAIN CARD
# =========================================================


st.markdown(
    """
    <div class="upload-heading">
        ☁️ Upload MRI Image
    </div>

    <div class="upload-text">
        Choose a JPG, JPEG, or PNG MRI image
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# =========================================================
# IMAGE + PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.write("")

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    st.write("")

    if st.button("🧠  Predict"):

        predicted_class, confidence = predict_image(
            image
        )

        # -----------------------------
        # RESULT
        # -----------------------------

        st.markdown(
            '<div class="result-heading">Prediction</div>',
            unsafe_allow_html=True
        )

        st.success(
            f"🧠  {predicted_class.upper()}"
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

# =========================================================
# CLOSE CARD
# =========================================================

st.markdown(
    "</div>",
    unsafe_allow_html=True
)

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Brain Tumor MRI Classification • CNN Model
    </div>
    """,
    unsafe_allow_html=True
)
