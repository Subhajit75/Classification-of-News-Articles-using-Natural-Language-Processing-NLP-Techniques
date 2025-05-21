import streamlit as st
import numpy as np
import gensim
import gdown
import re
import pytesseract
import requests
import os
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
from tensorflow.keras.models import load_model
from bs4 import BeautifulSoup

# Set Tesseract path (modify this if deploying on Linux or Cloud)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="News Article Category Classifier", layout="wide")

# Label map
label_map = {0: "🌍 World", 1: "🏅 Sports", 2: "💼 Business", 3: "🔬 Sci/Tech"}

# Download and load models
@st.cache_resource
def load_models():
    import gdown
    os.makedirs("models", exist_ok=True)

    # Correctly formatted Google Drive URLs with export=download
    rnn_url = "https://drive.google.com/uc?export=download&id=17aK4XwBbtejoawxoDbg4tyvGaKNsux6F"
    lstm_url = "https://drive.google.com/uc?export=download&id=1eIoSI4RFbNNEicdBPMUmt8z-23nbqztF"

    rnn_path = "models/news_classification_model_rnn.h5"
    lstm_path = "models/News_classification_model_LSTM_1.h5"

    try:
        if not os.path.exists(rnn_path):
            gdown.download(rnn_url, rnn_path, quiet=False, use_cookies=False)

        if not os.path.exists(lstm_path):
            gdown.download(lstm_url, lstm_path, quiet=False, use_cookies=False)
    except Exception as e:
        st.error(f"Failed to download models. Make sure your Drive files are public. Error: {e}")
        st.stop()

    return tf.keras.models.load_model(rnn_path), tf.keras.models.load_model(lstm_path)

# Download and load embeddings
@st.cache_resource
@st.cache_resource
def load_embeddings():
    import gdown
    os.makedirs("data", exist_ok=True)

    emb_url = "https://drive.google.com/uc?export=download&id=18vDkZalMnri75e9r7fefZB2oMtDaJLT9"
    emb_path = "data/numberbatch-en-19.08.txt"

    try:
        if not os.path.exists(emb_path):
            gdown.download(emb_url, emb_path, quiet=False, use_cookies=False)
    except Exception as e:
        st.error(f"Failed to download embeddings. Make sure your Drive file is public. Error: {e}")
        st.stop()

    return gensim.models.KeyedVectors.load_word2vec_format(emb_path, binary=False)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Create embedding vector
def get_embedding_vector(text, word_vectors, embedding_dim=300, max_length=200):
    tokens = text.split()
    embeddings = [word_vectors[word] for word in tokens if word in word_vectors]
    if not embeddings:
        return np.zeros((max_length, embedding_dim), dtype=np.float32)
    embeddings = np.array(embeddings[:max_length], dtype=np.float32)
    padding_needed = max_length - len(embeddings)
    if padding_needed > 0:
        embeddings = np.vstack([embeddings, np.zeros((padding_needed, embedding_dim), dtype=np.float32)])
    return embeddings

# Predict
def predict_category(text, rnn_model, lstm_model, word_vectors):
    tokens = text.split()
    embedding = get_embedding_vector(text, word_vectors)
    input_data = np.expand_dims(embedding, axis=0)
    model, name = (rnn_model, "RNN") if len(tokens) <= 20 else (lstm_model, "LSTM")
    predictions = model.predict(input_data)
    idx = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    return label_map[idx], name, confidence, predictions

# ---------- Streamlit UI ----------

st.title("📰 News Article Category Classifier")
st.markdown("Classify news from **text**, **PDF**, **image**, or **website** using AI models (RNN & LSTM) and ConceptNet embeddings.")

with st.expander("📖 How it works"):
    st.markdown("""
    - Input text is transformed into 300-dimensional word embeddings.
    - If the text has 20 words or fewer, we use a **Recurrent Neural Network (RNN)**.
    - For longer inputs, a **Long Short-Term Memory (LSTM)** model is used.
    - Models are trained on news headlines & description then predict one of four categories:
        - 🌍 World
        - 🏅 Sports
        - 💼 Business
        - 🔬 Sci/Tech
    """)

# Input mode
input_mode = st.radio("📥 Choose Input Type:", ["Text", "PDF", "Image", "URL"], horizontal=True)
input_text = ""

if input_mode == "Text":
    input_text = st.text_area("✍️ Enter news text:", height=200)

elif input_mode == "PDF":
    uploaded_pdf = st.file_uploader("📄 Upload PDF File", type=["pdf"])
    if uploaded_pdf:
        with st.spinner("Extracting text from PDF..."):
            pdf = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
            input_text = "\n".join([page.get_text() for page in pdf])

elif input_mode == "Image":
    uploaded_image = st.file_uploader("🖼️ Upload Image File", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)

        # Resize image to a smaller version (e.g., width=300 while maintaining aspect ratio)
        max_width = 300
        w_percent = (max_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        resized_image = image.resize((max_width, h_size), Image.Resampling.LANCZOS)

        st.image(resized_image, caption="Uploaded Image", use_container_width=False)

        with st.spinner("Performing OCR..."):
            input_text = pytesseract.image_to_string(image)

elif input_mode == "URL":
    url = st.text_input("🔗 Enter Website URL")
    if url:
        try:
            with st.spinner("Extracting text from website..."):
                response = requests.get(url)
                soup = BeautifulSoup(response.content, "html.parser")

                # Extract visible text from <p> tags
                paragraphs = soup.find_all("p")
                input_text = "\n".join(p.get_text() for p in paragraphs if p.get_text().strip())

                if not input_text.strip():
                    st.warning("⚠️ No readable content found on this page.")
        except Exception as e:
            st.error(f"❌ Failed to fetch content: {e}")

# Option to clean text
col1, col2 = st.columns([1, 1])
with col1:
    clean_option = st.checkbox("🧹 Clean and normalize input", value=True)
with col2:
    classify_btn = st.button("🚀 Classify")

# Classification
if classify_btn:
    if not input_text.strip():
        st.warning("⚠️ Please provide input text from one of the sources.")
    else:
        with st.spinner("🔍 Classifying..."):
            rnn_model, lstm_model = load_models()
            word_vectors = load_embeddings()
            processed_text = clean_text(input_text) if clean_option else input_text
            category, model_used, confidence, full_probs = predict_category(
                processed_text, rnn_model, lstm_model, word_vectors
            )

        st.success(f"### ✅ Predicted Category: {category}")
        st.info(f"🧠 Model Used: `{model_used}`")
        st.metric("📊 Confidence Score", f"{confidence*100:.2f}%")

        with st.expander("📈 Prediction Details"):
            for idx, prob in enumerate(full_probs[0]):
                st.write(f"{label_map[idx]}: {prob:.4f}")
