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


# ---------------------------- Model and Label Mapping -----------------------------

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


# ---------------------------- Embedding Download + Load -----------------------------

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

st.title("📰 News Article Category Classifier")  # Initialize Streamlit app with title and description
st.markdown("Classify news from **text**, **PDF**, **image**, or **website** using AI models (RNN & LSTM) and ConceptNet embeddings.") # Provide a brief description of the app's functionality
with st.expander("📖 How it works"): # Add a sidebar for navigation
    # Provide a brief explanation of how the classification works
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
    input_text = st.text_area("✍️ Enter news text:", height=200) # Text area for user to input news text

elif input_mode == "PDF": # PDF input mode
    uploaded_pdf = st.file_uploader("📄 Upload PDF File", type=["pdf"])  # File uploader for PDF files
    if uploaded_pdf: # If a PDF file is uploaded
        with st.spinner("Extracting text from PDF..."): # Show a spinner while processing
            pdf = fitz.open(stream=uploaded_pdf.read(), filetype="pdf") # Open the PDF file using PyMuPDF
            input_text = "\n".join([page.get_text() for page in pdf]) # Extract text from each page and join them into a single string

elif input_mode == "Image": # Image input mode
    uploaded_image = st.file_uploader("🖼️ Upload Image File", type=["jpg", "jpeg", "png"]) # File uploader for image files
    if uploaded_image: # If an image file is uploaded
        image = Image.open(uploaded_image) # Open the image using Pillow

        max_width = 300 # Resize image to a smaller version (e.g., width=300 while maintaining aspect ratio)
        w_percent = (max_width / float(image.size[0])) # Calculate the width percentage for resizing
        h_size = int((float(image.size[1]) * float(w_percent))) # Calculate the new height based on the width percentage
        resized_image = image.resize((max_width, h_size), Image.Resampling.LANCZOS) # Resize the image using Lanczos resampling
        st.image(resized_image, caption="Uploaded Image", use_container_width=False) # Display the resized image in the app
        with st.spinner("Performing OCR..."): # Show a spinner while performing OCR
            input_text = pytesseract.image_to_string(image) # Use Tesseract to extract text from the image

elif input_mode == "URL": # URL input mode
    url = st.text_input("🔗 Enter Website URL")  # Text input for website URL
    if url: # If a URL is provided
        try: # Fetch the content of the URL 
            with st.spinner("Extracting text from website..."): # Show a spinner while fetching content
                response = requests.get(url) # Check if the URL is valid and fetch its content
                soup = BeautifulSoup(response.content, "html.parser")
                # Extract text from all paragraph tags in the HTML content
                paragraphs = soup.find_all("p") # Find all paragraph tags in the HTML content
                input_text = "\n".join(p.get_text() for p in paragraphs if p.get_text().strip()) # Join the text from all paragraphs into a single string

                if not input_text.strip(): # If no readable content is found
                    st.warning("⚠️ No readable content found on this page.")  # Display a warning message
        except Exception as e: # Handle any exceptions that occur while fetching the URL
            st.error(f"❌ Failed to fetch content: {e}") # Display an error message if fetching fails


# ------------------------- Options Section -----------------------------------

col1, col2 = st.columns([1, 1]) # Create two columns for options
with col1: # Checkbox to clean and normalize input text
    clean_option = st.checkbox("🧹 Clean and normalize input", value=True)  # Option to clean text
with col2: # Button to clear input text
    classify_btn = st.button("🚀 Classify") # Clear input button


# ------------------------- Classification Section -----------------------------------

if classify_btn: # If the classify button is clicked
    if not input_text.strip(): # Check if input text is empty
        st.warning("⚠️ Please provide input text from one of the sources.") # Display a warning if no input text is provided
    else: # If input text is provided
        with st.spinner("🔍 Classifying..."): # Show a spinner while classifying the input text
            rnn_model, lstm_model = load_models() # Load the pre-trained RNN and LSTM models
            word_vectors = load_embeddings() # Load the pre-trained word embeddings
            processed_text = clean_text(input_text) if clean_option else input_text # Clean the input text if the clean option is selected
            # Predict the category using the selected model based on the length of the input text
            category, model_used, confidence, full_probs = predict_category( 
                processed_text, rnn_model, lstm_model, word_vectors 
            )
# ------------------------- Display Results -----------------------------------

        st.success(f"### ✅ Predicted Category: {category}") # Display the predicted category
        st.info(f"🧠 Model Used: `{model_used}`")  # Display the model used for classification and the confidence score
        st.metric("📊 Confidence Score", f"{confidence*100:.2f}%")  # Display the confidence score as a percentage


# ------------------------- Prediction Details -----------------------------------

        with st.expander("📈 Prediction Details"): # Display detailed prediction probabilities for each category
            for idx, prob in enumerate(full_probs[0]): # Iterate through the prediction probabilities
                st.write(f"{label_map[idx]}: {prob:.4f}") # Display the category and its corresponding probability


