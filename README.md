# 🎬 IMDB Sentiment Analysis - SimpleRNN Project

A machine learning project that uses a **SimpleRNN (Recurrent Neural Network)** to predict the sentiment of movie reviews from the IMDB dataset. The project includes both Jupyter notebooks for model training/analysis and a **Streamlit web application** for interactive predictions.

## 📋 Project Overview

This project demonstrates:
- Text preprocessing and embedding for NLP tasks
- Building and training a SimpleRNN model with Keras/TensorFlow
- Sentiment analysis (Positive/Negative classification)
- Model evaluation and prediction
- Web app deployment using Streamlit

**Model Performance**: ~79% accuracy on test data

## 📁 Project Structure

```
simpleRNN/
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies
├── streamlit_app.py                 # Interactive web application
│
├── predict.ipynb                    # Main prediction notebook with examples
├── embedding.ipynb                  # Embedding layer exploration
├── predictions.ipynb                # Additional prediction examples
├── simpleRNN.ipynb                  # Model architecture & training
│
├── simple_rnn_imdb.h5              # Pre-trained model (main)
├── simple_rnn_imdb (1).h5          # Alternative pre-trained model
```

## 🔧 Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Streamlit**: Web application framework
- **Python 3.11+**: Programming language

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd simpleRNN
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- tensorflow >= 2.15.0
- keras >= 2.15.0
- streamlit >= 1.50.0
- numpy >= 1.26.0
- pandas >= 2.0.0

## 🚀 Quick Start

### Option 1: Run the Streamlit Web App (Recommended for Users)
```bash
streamlit run streamlit_app.py
```
Then open your browser to `http://localhost:8501`

**Features:**
- 📝 Enter custom movie reviews
- 🔮 Get real-time sentiment predictions
- 📊 View confidence scores
- ✨ Try pre-built positive/negative examples
- ⚙️ Adjust confidence threshold

### Option 2: Use Jupyter Notebooks (For Development/Analysis)

1. **Main Prediction Notebook**: `predict.ipynb`
   - Load pre-trained model
   - Test on custom reviews
   - Evaluate on IMDB test data
   - Examples of positive/negative predictions

2. **Embedding Notebook**: `embedding.ipynb`
   - Explore word embeddings
   - Visualize embedding layer

3. **Additional Analysis**: `predictions.ipynb`, `simpleRNN.ipynb`
   - Model training details
   - Architecture visualization

## 📊 Model Architecture

```
Input (Variable length text)
    ↓
[Embedding Layer] - 10,000 vocab size, 128 dimensions
    ↓
[SimpleRNN Layer] - 128 units, ReLU activation
    ↓
[Dense Layer] - 1 unit, Sigmoid activation
    ↓
Output (0.0 to 1.0 probability)
    ↓
Classification: Negative (< 0.5) or Positive (≥ 0.5)
```

## 🎯 How It Works

### Text Preprocessing
1. Convert text to lowercase
2. Split into words
3. Map words to indices using IMDB word index (max 10,000)
4. Handle out-of-vocabulary (OOV) words with token 2
5. Pad sequences to max length of 500 tokens

### Prediction
1. Preprocess user input text
2. Pass through embedding layer (converts indices to 128-dim vectors)
3. Process through SimpleRNN layer (captures sequential patterns)
4. Dense layer produces probability score
5. Classification: score > 0.5 = Positive, score ≤ 0.5 = Negative

### Uncertainty Detection
- Predictions within ±0.1 of threshold (0.4-0.6) are marked as "UNCERTAIN"
- Indicates model's low confidence in the prediction

## 📈 Model Performance

- **Accuracy on Test Data**: ~79%
- **Loss**: ~0.45 (Binary Crossentropy)
- **Training Data**: IMDB movie review dataset (25,000 reviews)
- **Vocab Size**: 10,000 most common words

## 📝 Example Usage

### Via Streamlit App
1. Click "✨ Try Positive Example" or "😞 Try Negative Example"
2. Review text appears in text area
3. Click "🔮 Predict Sentiment"
4. View results with confidence score

### Via Python Code
```python
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model

# Load model and word index
model = load_model('simple_rnn_imdb(1).h5')
word_index = imdb.get_word_index()

# Preprocess text
review = "This movie was amazing!"
# ... preprocessing steps ...

# Predict
prediction = model.predict(preprocessed_text)
sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
```

## 🔍 Understanding Predictions

| Confidence | Sentiment | Meaning |
|-----------|-----------|---------|
| 0.90-1.00 | Positive | Strong positive sentiment |
| 0.60-0.89 | Positive | Moderate positive sentiment |
| 0.50-0.59 | UNCERTAIN | Model is unsure |
| 0.41-0.49 | UNCERTAIN | Model is unsure |
| 0.10-0.40 | Negative | Moderate negative sentiment |
| 0.00-0.09 | Negative | Strong negative sentiment |

## ⚠️ Limitations

1. **Model Accuracy**: 79% - not perfect, some predictions will be wrong
2. **Vocabulary Limited**: Only recognizes 10,000 most common IMDB words
3. **Short Reviews**: Works best with reviews of reasonable length (50-500 tokens)
4. **IMDB-Specific**: Trained on movie reviews, may not work well on other domains
5. **Binary Classification**: Only Positive/Negative, no neutral sentiment

## 🛠️ Troubleshooting

### Model File Not Found
- Ensure `simple_rnn_imdb(1).h5` is in the same directory
- Check working directory with `os.getcwd()` in notebooks

### TensorFlow Import Error
```bash
pip install --upgrade tensorflow keras
```

### Streamlit Not Starting
```bash
pip install streamlit
streamlit run streamlit_app.py --logger.level=debug
```

### Wrong Predictions
- This is normal (~21% error rate)
- Model uncertainty is flagged when confidence is near 0.5
- Consider retraining with more epochs for better accuracy

## 📚 Training the Model (Optional)

To retrain the model with more epochs:

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import tensorflow as tf

# Load data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Preprocess
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)

# Build and train model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 128, input_length=500),
    tf.keras.layers.SimpleRNN(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model.save('simple_rnn_imdb_retrained.h5')
```

## 🔗 Resources

- [Keras Documentation](https://keras.io/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [IMDB Dataset](https://www.imdb.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Created as a machine learning demo project for sentiment analysis using SimpleRNN architecture.

## 🤝 Contributing

Feel free to fork, modify, and improve this project!

---

**Last Updated**: December 2025
**Python Version**: 3.11+
**Status**: Active
