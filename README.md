# 🧠 Sentiment Analysis Project

This project implements a **machine learning-based sentiment analysis system** that classifies text into three categories:

- **Positive**
- **Neutral**
- **Negative**

The model is trained using traditional NLP techniques and deployed via a Python class for easy inference.

---

## 📁 Project Structure

```

.
├── sentiment.py                    # Model inference script
├── training.ipynb                  # Model training notebook
├── logistic_regression_model.pkl
├── tfidf_vectorizor.pkl
├── label_encoder.pkl    

````

---

## ⚙️ Setup Instructions (Virtual Environment)

### 1. Create Virtual Environment

```bash
python -m venv venv
````

### 2. Activate Virtual Environment

* **Windows:**

```bash
venv\Scripts\activate.ps1
```

* **Mac/Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r huggingface_hub ipywidgets fastparquet wordcloud nltk pandas seaborn matplotlib scikit-learn
```

### 4. Download NLTK Resources

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## 🏋️ Model Training (`training.ipynb`)

The `training.ipynb` notebook contains the complete pipeline for training the sentiment model.

### Steps Performed:

1. **Data Loading**
2. **Text Preprocessing**

   * Lowercasing
   * Removing special characters
   * Tokenization
   * Stopword removal
3. **Feature Extraction**

   * TF-IDF Vectorization
4. **Model Training**

   * Logistic Regression classifier
5. **Model Saving**

   * `logistic_regression_model.pkl`
   * `tfidf_vectorizor.pkl`

### How to Use:

1. Open the notebook:

   ```bash
   jupyter notebook training.ipynb
   ```
2. Run all cells sequentially.
3. Ensure `.pkl` files are generated after training.

---

## 🚀 Model Usage (`sentiment.py`)

The `sentiment.py` file is responsible for **loading the trained model and making predictions**.

### Key Features:

* Loads trained model and vectorizer
* Cleans input text
* Predicts sentiment

### Example Usage:

```python
from sentiment import SentimentAnalyser

analyzer = SentimentAnalyser()

text = "This product is amazing!"
print(analyzer.predict(text))
```

### Output:

```
Positive
```

---

## 🔍 How It Works

1. Input text is cleaned:

   * Removes special characters
   * Converts to lowercase
   * Removes stopwords
2. Text is transformed using TF-IDF vectorizer
3. Logistic Regression model predicts class
4. Output is mapped to:

   * `0 → Negative`
   * `1 → Neutral`
   * `2 → Positive`

---

## 📊 Results

The model was evaluated on a test set of **5000 samples** using standard classification metrics.

### 🔢 Classification Report


```
                precision    recall  f1-score   support

           0       0.79      0.73      0.76      2000
           1       0.43      0.53      0.47      1000
           2       0.84      0.79      0.81      2000

accuracy                               0.72      5000
macro avg          0.68      0.69      0.68      5000
weighted avg       0.74      0.72      0.72      5000

```

---

### 📌 Key Insights

- **Overall Accuracy:** `72%`
- **Best Performing Class:**  
  - **Positive (Class 2)** with F1-score of **0.81**
- **Strong Performance:**  
  - **Negative (Class 0)** with F1-score of **0.76**
- **Weakest Area:**  
  - **Neutral (Class 1)** with F1-score of **0.47**

---

### 📈 Analysis

- The model performs **well on polarized sentiments** (positive & negative).
- Performance drops on **neutral class**, likely due to:
  - Ambiguous language in neutral reviews
  - Class overlap between positive and negative sentiments
- Slight class imbalance (2000 / 1000 / 2000) may also impact performance.

---

## 📦 Dependencies

Key libraries used:

* `scikit-learn`
* `nltk`
* `pandas`
* `matplotlib`
* `seaborn`
* `wordcloud`
* `huggingface_hub`

---

## 📚 Dataset Citation

This project uses the following dataset:

👉 [https://huggingface.co/datasets/mteb/AmazonReviewsClassification](https://huggingface.co/datasets/mteb/AmazonReviewsClassification)

---

## 👨‍💻 Author

Developed as part of a sentiment analysis ML project.