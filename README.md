# 🧠 Sentiment Analysis Full-Stack Application

A modern full-stack sentiment analysis application built with **FastAPI** (backend) and **Next.js** (frontend) that classifies text reviews into Positive, Neutral, or Negative sentiments using a trained Logistic Regression model.

> **Note:** This README documents the **full-stack branch**. The main branch contains the standalone terminal version of the project.

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Backend Setup](#-backend-setup)
- [Frontend Setup](#-frontend-setup)
- [API Documentation](#-api-documentation)
- [Testing](#-testing)

---

## ✨ Features

### Backend (FastAPI)
- 🚀 High-performance REST API with automatic OpenAPI documentation
- ✅ CORS enabled for seamless frontend communication
- 🔒 Input validation with Pydantic models
- 📊 Health check endpoint for monitoring
- ⚡ Pre-loaded sentiment analyzer for instant responses

### Frontend (Next.js)
- 🎨 Modern, responsive UI with gradient design
- 📱 Mobile-friendly layout
- ⚡ Real-time sentiment analysis with loading states
- 🎯 Example reviews for quick testing
- 🎨 Color-coded sentiment results (Positive/Negative/Neutral)
- ✨ Smooth animations and transitions

### ML Model
- 🧠 Logistic Regression classifier trained on Amazon reviews
- 📊 72% overall accuracy
- 🎯 Best performance on Positive (F1: 0.81) and Negative (F1: 0.76) sentiments
- 📝 Text preprocessing with TF-IDF vectorization

---

## 📁 Project Structure

```
sentiment-analysis-fullstack/
├── backend/                          # FastAPI Backend
│   ├── main.py                       # FastAPI application with REST endpoints
│   ├── model.py                      # SentimentAnalyser class for ML inference
│   ├── requirements.txt              # Python dependencies
│   ├── logistic_regression_model.pkl # Trained ML model
│   ├── tfidf_vectorizor.pkl          # TF-IDF vectorizer
│   └── label_encoder.pkl             # Label encoder (if applicable)
│
├── frontend/                         # Next.js Frontend
│   ├── package.json                  # Node.js dependencies
│   ├── next.config.js                # Next.js configuration
│   ├── .env.local                    # Environment variables (optional)
│   └── src/
│       ├── app/                      # App Router structure
│       │   ├── layout.js             # Root layout with metadata
│       │   ├── page.js               # Main page component
│       │   └── globals.css           # Global styles
│       └── components/
│           └── SentimentAnalyzer.js  # Main client component
│
├── .gitignore                        # Git ignore rules
├── README.md                         # This file
└── fullstack-branch/                 # Documentation for full-stack branch
```

### Key Files Explained

#### Backend Files

| File | Description |
|------|-------------|
| `main.py` | FastAPI application with REST endpoints, CORS middleware, and Pydantic models |
| `model.py` | `SentimentAnalyser` class that loads ML model and performs sentiment predictions |
| `requirements.txt` | Python package dependencies (FastAPI, uvicorn, scikit-learn, nltk, etc.) |
| `logistic_regression_model.pkl` | Serialized Logistic Regression model |
| `tfidf_vectorizor.pkl` | Serialized TF-IDF vectorizer for text preprocessing |

#### Frontend Files

| File | Description |
|------|-------------|
| `page.js` | Main page component that renders the SentimentAnalyzer |
| `SentimentAnalyzer.js` | Client component with form, API calls, and result display |
| `globals.css` | Global CSS with gradient backgrounds and responsive design |
| `layout.js` | Root layout with Inter font and metadata |

---

## 🛠️ Prerequisites

Before setting up the project, ensure you have the following installed:

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| [Python](https://www.python.org/downloads/) | 3.8+ | Backend runtime |
| [Node.js](https://nodejs.org/) | 18+ | Frontend runtime |
| [npm](https://www.npmjs.com/) | 9+ | Package manager for frontend |
| [Git](https://git-scm.com/) | Latest | Version control |

### Optional but Recommended

- **[VS Code](https://code.visualstudio.com/)** with Python and ESLint extensions
- **[Postman](https://www.postman.com/)** or **[curl](https://curl.se/)** for API testing

---

## 🚀 Quick Start

The fastest way to get started:

```bash
# Clone the repository
git clone -b full-stack https://github.com/sherrytelli/amazon-review-sentiment-analysis.git

cd sentiment-analysis-fullstack
```

Navigate to `http://localhost:3000` in your browser.

### 🔧 Backend Setup

### Step 1: Navigate to Backend Directory

```bash
cd backend
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> **Verify activation:** Your terminal prompt should show `(venv)` prefix.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `nltk` - Natural language processing
- `scikit-learn` - Machine learning (model inference)
- `numpy` - Numerical computing

### Step 4: Download NLTK Resources

The sentiment analyzer requires NLTK data for text preprocessing. Run this Python script:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

Or run it directly from the command line:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```
### Step 5: Run the Backend Server

```bash
fastapi dev main.py
```

### Backend URLs

| Endpoint | Description |
|----------|-------------|
| `http://localhost:8000` | API base URL |
| `http://localhost:8000/docs` | Swagger UI (interactive API docs) |
| `http://localhost:8000/redoc` | ReDoc (alternative API docs) |
| `http://localhost:8000/health` | Health check endpoint |

---

### 🎨 Frontend Setup

### Step 1: Navigate to Frontend Directory

```bash
cd frontend
```

### Step 2: Install Dependencies

```bash
npm install
```

This will install:
- `next` - React framework
- `react` & `react-dom` - React libraries
- Development dependencies (ESLint, etc.)

### Step 3: Configure Environment Variables (Optional)

Create a `.env.local` file in the `frontend/` directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

> **Note:** If not set, the app defaults to `http://localhost:8000`.

### Step 4: Run the Development Server

```bash
npm run dev
```

The development server will start with output like:

```
ready - started server on 0.0.0.0:3000, url: http://localhost:3000
info  - You are using Node.js 18.17.0
event - compiled client and server successfully in 1234 ms
```

### Step 5: Access the Application

Open your browser and navigate to:

```
http://localhost:3000
```

You should see the Sentiment Analysis application with:
- A text input area for entering reviews
- An "Analyze Sentiment" button
- Example review buttons for quick testing
- Results displayed with color-coded sentiment indicators

---

## 📡 API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### 1. Health Check

**GET** `/health`

Check if the API and sentiment analyzer are running properly.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true
}
```

---

#### 2. Analyze Sentiment (POST)

**POST** `/analyze`

Analyze the sentiment of a text review.

**Request Body:**
```json
{
  "text": "This product is absolutely amazing! Best purchase ever.",
  "review_id": "optional-unique-id"
}
```

**Response:**
```json
{
  "review_id": "optional-unique-id",
  "text": "This product is absolutely amazing! Best purchase ever.",
  "sentiment": "Positive"
}
```

**Possible Sentiments:**
- `"Positive"` - The review expresses positive sentiment
- `"Neutral"` - The review is neutral or ambiguous
- `"Negative"` - The review expresses negative sentiment

**Error Responses:**

| Status Code | Error | Description |
|-------------|-------|-------------|
| 400 | `Text field cannot be empty` | Empty text provided |
| 500 | `Error analyzing sentiment: ...` | Internal error during analysis |
| 503 | `Sentiment analyzer not available` | Models failed to load |

---

#### 3. Analyze Sentiment (GET)

**GET** `/analyze/text/{text}`

Alternative endpoint for quick testing via URL.

**Example:**
```
GET /analyze/text/This%20product%20is%20amazing
```

**Response:**
```json
{
  "text": "This product is amazing",
  "sentiment": "Positive"
}
```

> **Note:** URL-encode the text parameter. For complex texts, prefer the POST endpoint.

---

### Testing with cURL

**POST Request:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product! It works perfectly.", "review_id": "test-1"}'
```

**GET Request:**
```bash
curl "http://localhost:8000/analyze/text/I%20love%20this%20product"
```

**Health Check:**
```bash
curl "http://localhost:8000/health"
```

### Testing with Python

```python
import requests

# POST request
response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "text": "This product is absolutely amazing!",
        "review_id": "python-test-1"
    }
)

print(response.json())
# {'review_id': 'python-test-1', 'text': 'This product is absolutely amazing!', 'sentiment': 'Positive'}

# GET request
response = requests.get("http://localhost:8000/health")
print(response.json())
# {'status': 'healthy', 'version': '1.0.0', 'models_loaded': True}
```

---

## 📊 Model Performance

The underlying ML model has the following performance characteristics:

| Metric | Negative | Neutral | Positive | Overall |
|--------|----------|---------|----------|---------|
| Precision | 0.79 | 0.43 | 0.84 | - |
| Recall | 0.73 | 0.53 | 0.79 | - |
| F1-Score | 0.76 | 0.47 | 0.81 | 0.72 |

**Key Insights:**
- ✅ Best performance on **Positive** sentiments (F1: 0.81)
- ✅ Strong performance on **Negative** sentiments (F1: 0.76)
- ⚠️ Weakest performance on **Neutral** sentiments (F1: 0.47)
- 📊 Overall accuracy: **72%**

**Recommendations for Improvement:**
- Collect more neutral examples for training
- Try ensemble methods (Random Forest, XGBoost)
- Experiment with deeper neural networks
- Add data augmentation techniques

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 👨‍💻 Author

Developed as part of a sentiment analysis ML project.

**Full-Stack Implementation:**
- Backend: FastAPI + Python
- Frontend: Next.js + React
- ML Model: Logistic Regression with TF-IDF

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Dataset Source](https://huggingface.co/datasets/mteb/AmazonReviewsClassification)

---

## 🆘 Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#-troubleshooting) section above
2. Review the [API Documentation](#-api-documentation)
3. Check browser console for frontend errors
4. Check terminal output for backend errors
5. Open an issue on the repository

---

## 📝 Changelog

### Version 1.0.0 (Full-Stack Branch)
- ✅ Added FastAPI backend with REST endpoints
- ✅ Added Next.js frontend with responsive UI
- ✅ Integrated sentiment analysis model
- ✅ Added CORS middleware
- ✅ Added health check endpoint
- ✅ Added interactive API documentation
- ✅ Added example reviews for testing
- ✅ Added error handling and validation
- ✅ Added loading states and animations

---