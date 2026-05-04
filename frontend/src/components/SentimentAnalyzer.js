'use client'

import { useState } from 'react'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function SentimentAnalyzer() {
  const [reviewText, setReviewText] = useState('')
  const [sentiment, setSentiment] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const sampleReviews = [
    "This product is absolutely amazing! Best purchase I've ever made.",
    "It's okay, nothing special but does the job.",
    "Terrible quality! Completely disappointed with this purchase.",
    "Great customer service and fast shipping. Highly recommend!",
    "Average product, expected better for the price."
  ]

  const analyzeSentiment = async (text) => {
    if (!text.trim()) {
      setError('Please enter some text to analyze')
      return
    }

    setLoading(true)
    setError(null)
    setSentiment(null)

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          review_id: Date.now().toString()
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to analyze sentiment')
      }

      const data = await response.json()
      setSentiment(data)
    } catch (err) {
      setError(err.message || 'An error occurred while analyzing the sentiment')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    await analyzeSentiment(reviewText)
  }

  const handleExampleClick = (review) => {
    setReviewText(review)
    analyzeSentiment(review)
  }

  const getSentimentClass = (sentiment) => {
    if (!sentiment) return ''
    return sentiment.toLowerCase()
  }

  const getSentimentEmoji = (sentiment) => {
    switch (sentiment) {
      case 'Positive':
        return '😊'
      case 'Neutral':
        return '😐'
      case 'Negative':
        return '😞'
      default:
        return '🤔'
    }
  }

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="review">Enter your review:</label>
          <textarea
            id="review"
            value={reviewText}
            onChange={(e) => setReviewText(e.target.value)}
            placeholder="Type your review here..."
            required
          />
        </div>
        
        <button type="submit" disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze Sentiment'}
        </button>
      </form>

      {loading && <div className="loading show">⏳ Analyzing your review...</div>}

      {error && (
        <div className="error-message show">
          ❌ {error}
        </div>
      )}

      {sentiment && (
        <div className={`result-container show ${getSentimentClass(sentiment.sentiment)}`}>
          <div className="sentiment-label">
            {getSentimentEmoji(sentiment.sentiment)} {sentiment.sentiment}
          </div>
          <div className="review-text">
            "{sentiment.text}"
          </div>
        </div>
      )}

      <div className="examples">
        <h3>📝 Try these example reviews:</h3>
        <div className="example-buttons">
          {sampleReviews.map((review, index) => (
            <button
              key={index}
              className="example-btn"
              onClick={() => handleExampleClick(review)}
              disabled={loading}
            >
              {review.length > 50 ? review.substring(0, 50) + '...' : review}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

