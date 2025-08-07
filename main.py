import streamlit as st  # Streamlit for web app
import numpy as np      # NumPy for numerical operations
import gensim           # Gensim for word embeddings 
import gdown            # gdown for downloading files from Google Drive
import re               # Regular expressions for text cleaning
import pytesseract      # Tesseract for OCR
import requests         # Requests for HTTP requests
import os               # OS for file path operations
import gzip             # Gzip for handling compressed files
import shutil           # Shutil for file operations
from io import BytesIO  # BytesIO for in-memory file handling
from PIL import Image   # Pillow for image processing
import fitz             # PyMuPDF for PDF text extraction
import tensorflow as tf # TensorFlow for deep learning models
from bs4 import BeautifulSoup   # BeautifulSoup for HTML parsing


# ---------------------------- Setup ---------------------------------------------

# Set Tesseract path (modify if on Linux or Cloud)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # Change this path to your local Tesseract installation

st.set_page_config(page_title="News Article Category Classifier", layout="wide")     # Set Streamlit page title and layout


# ---------------------------- Model and Label Mapping ----------------------------

# Define label mapping for categories
# These labels correspond to the indices of the model's output layer
label_map = {0: "🌍 World", 1: "🏅 Sports", 2: "💼 Business", 3: "🔬 Sci/Tech"} 

# Load models directly from GitHub repo (already uploaded .h5 files)
@st.cache_resource  # Cache model loading to avoid reloading every time
def load_models():  # Load pre-trained RNN and LSTM models
    # Define paths for the models
    # These paths should point to the .h5 files in your GitHub repository
    rnn_path = "models/news_classification_model_rnn.h5" # Change this path to your local RNN model file
    lstm_path = "models/News_classification_model_LSTM_1.h5" # Change this path to your local LSTM model file
    return tf.keras.models.load_model(rnn_path), tf.keras.models.load_model(lstm_path) # Load and return the models


# ---------------------------- Embedding Download + Load ----------------------------

@st.cache_resource     # Cache embeddings loading to avoid reloading every time
def load_embeddings(): # Load pre-trained word embeddings
    emb_path = r"D:\Project_Env\News Article classification\News Article classification\data\numberbatch-en-19.08.txt" # Change this path to your local embedding file

    if not os.path.exists(emb_path):  # Check if the embedding file exists
        st.error("Embedding file not found at the specified path.") # Provide a clear error message
        st.stop()   # Stop execution if file not found
    
    return gensim.models.KeyedVectors.load_word2vec_format(emb_path, binary=False)  # Load embeddings in text format


# ---------------------------- Text Processing Functions ----------------------------

def clean_text(text): # Function to clean and normalize input text
    text = text.lower() # Convert text to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text) # Remove non-alphabetic characters
    text = re.sub(r"\s+", " ", text).strip() # Remove extra spaces and strip leading/trailing whitespace
    return text # Normalize text by removing punctuation and extra spaces


# ---------------------------- Embedding Vector Creation ----------------------------

def get_embedding_vector(text, word_vectors, embedding_dim=300, max_length=200): # Function to convert text into a fixed-size embedding vector
    tokens = text.split() # Split text into tokens (words)
    embeddings = [word_vectors[word] for word in tokens if word in word_vectors] # Get embeddings for each token if it exists in the word vectors
    if not embeddings: # If no embeddings found, return a zero vector
        return np.zeros((max_length, embedding_dim), dtype=np.float32) # If no embeddings, return zero vector
    embeddings = np.array(embeddings[:max_length], dtype=np.float32) # Convert list of embeddings to a NumPy array
    padding_needed = max_length - len(embeddings) 
    if padding_needed > 0: # If the number of embeddings is less than max_length, pad with zeros
        embeddings = np.vstack([embeddings, np.zeros((padding_needed, embedding_dim), dtype=np.float32)]) # Pad with zeros to reach max_length
    return embeddings # np.expand_dims(embeddings, axis=0)  # Add batch dimension for model input


# ---------------------------- Prediction Function ----------------------------

def predict_category(text, rnn_model, lstm_model, word_vectors): # Function to predict the category of the input text
    tokens = text.split() # Split text into tokens
    embedding = get_embedding_vector(text, word_vectors) # Get the embedding vector for the input text
    input_data = np.expand_dims(embedding, axis=0) # Add batch dimension for model input
    model, name = (rnn_model, "RNN") if len(tokens) <= 20 else (lstm_model, "LSTM") # Choose model based on token count
    predictions = model.predict(input_data) # Make predictions using the selected model
    idx = np.argmax(predictions, axis=1)[0] # Get the index of the highest predicted category
    confidence = float(np.max(predictions)) # Get the confidence score of the prediction
    return label_map[idx], name, confidence, predictions # Return the predicted category, model name, confidence score, and full prediction probabilities

# ------------------------- Streamlit UI -----------------------------------

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

# ------------------------- Input Section ----------------------------------- 

input_mode = st.radio("📥 Choose Input Type:", ["Text", "PDF", "Image", "URL"], horizontal=True) # Radio buttons for selecting input type
input_text = ""  # Initialize input text variable

if input_mode == "Text": # Text input mode
    input_text = st.text_area("✍️ Enter news text:", height=200) # Text area for user input

elif input_mode == "PDF": # PDF input mode 
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
