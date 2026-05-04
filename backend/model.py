import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

class SentimentAnalyser:
    def __init__(self):
        # Loading model
        with open("logistic_regression_model.pkl", "rb") as f:
            self.__model = pickle.load(f)
            
        # Loading vectorizer
        with open("tfidf_vectorizor.pkl", "rb") as f:
            self.__vectorizor = pickle.load(f)
            
        self.__sentiment_mapping = {
            0: "Negative", 
            1: "Neutral",
            2: "Positive",
        }
        
        self.__stop_words = stopwords.words("english")
        
    def __clean_text(self, text: str):
        # Removing newline and tab characters
        text = text.replace('\n', ' ').replace('\t', ' ')
        
        # Lower casing text
        text = text.lower()
        
        # Removing any character that is not alphanumeric
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Tokenizing the text
        words = word_tokenize(text)
        
        # Cleaning the text
        cleaned_words = [word for word in words if word not in self.__stop_words]
        
        # Returning cleaned text
        return " ".join(cleaned_words)
    
    def predict(self, text: str) -> str:
        # Cleaning text
        cleaned_text = self.__clean_text(text)
        
        # Vectorizing text
        vectorized_text = self.__vectorizor.transform([cleaned_text])
        
        # Making prediction
        prediction = self.__model.predict(vectorized_text)
        
        # Returning sentiment
        return self.__sentiment_mapping[prediction[0]]

