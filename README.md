


# News Article Category Classifier 🌍🏅💼🔬


A multi-input AI system that classifies news articles into 4 categories using RNN/LSTM models with ConceptNet embeddings. Supports text, PDFs, images (OCR), and website URLs.

## Features ✨

- **Multi-model architecture**: 
  - RNN for short texts (≤20 words)
  - LSTM for longer articles
- **Multiple input formats**:
  - 📝 Direct text input
  - 📄 PDF document parsing
  - 🖼️ Image OCR (Tesseract)
  - 🌐 Web scraping (URL)
- **ConceptNet NumberBatch** embeddings (300D)
- **Clean text preprocessing** with regex
- **Confidence scores** and detailed prediction metrics
- **Streamlit** web interface

## Categories 🏷️

1. 🌍 World News
2. 🏅 Sports
3. 💼 Business
4. 🔬 Science/Technology

## Installation ⚙️

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/news-classifier.git
   cd news-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR (Windows):
   - Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add to PATH or specify location in code

## Usage 🚀

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default browser at `localhost:8501`.

## How It Works 🧠

1. **Text Processing**:
   - Input is cleaned (lowercase, special chars removed)
   - Words converted to ConceptNet NumberBatch embeddings

2. **Model Selection**:
   - RNN for short texts (≤20 words)
   - LSTM for longer articles

3. **Prediction**:
   - Models output probabilities for each category
   - Highest probability determines final classification

## File Structure 📂

```
news-classifier/
├── app.py                # Main Streamlit application
├── models/               # Pretrained models (RNN/LSTM)
│   ├── news_classification_model_rnn.h5
│   └── News_classification_model_LSTM_1.h5
├── data/                 # Embeddings
│   └── numberbatch-en-19.08.txt
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Dependencies 📦

- Python 3.7+
- Streamlit
- TensorFlow 2.x
- Gensim
- PyMuPDF (for PDF)
- Pytesseract (for OCR)
- BeautifulSoup4 (for web)
- Requests

## Limitations ⚠️

- English language only
- Best performance on news headlines/short articles
- OCR accuracy depends on image quality
- Web scraping may fail on JS-heavy sites

## Contributing 🤝

Pull requests are welcome! For major changes, please open an issue first.

## License 📜

MIT License
```

This README includes:
- Badges for key technologies
- Clear feature list
- Installation instructions
- Usage guide
- Technical explanation
- File structure
- Dependencies
- Limitations
- Contribution guidelines

You may want to:
1. Replace placeholder GitHub URL
2. Add your own license if not using MIT
3. Add deployment instructions if hosting online
4. Include screenshots by adding an `images/` folder
