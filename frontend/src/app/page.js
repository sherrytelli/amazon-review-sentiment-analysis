import SentimentAnalyzer from '@/components/SentimentAnalyzer'

export default function Home() {
  return (
    <div className="container">
      <h1>Sentiment Analysis</h1>
      <p className="subtitle">Enter a review to analyze its sentiment</p>
      <SentimentAnalyzer />
    </div>
  )
}

