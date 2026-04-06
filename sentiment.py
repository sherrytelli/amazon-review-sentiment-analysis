import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

class SentimentAnalyser:
    def __init__(self):
        #loading model
        with open("logistic_regression_model.pkl", "rb") as f:
            self.__model =  pickle.load(f)
            
        #loading vectorizor
        with open("tfidf_vectorizor.pkl", "rb") as f:
            self.__vectorizor = pickle.load(f)
            
        self.__sentiment_mapping = {
                0: "Negative", 
                1: "Neutral",
                2: "Positive",
            }
        
        self.__stop_words = stopwords.words("english")
        
    def __clean_text(self, text: str):
        #removing newline and tab characters
        text = text = text.replace('\n', ' ').replace('\t', ' ')
        
        #lower casing text
        text = text.lower()
        
        #removing any character that is not alphanumeric
        text = text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        #tokeinzing the text
        words = word_tokenize(text)
        
        #cleaning the text
        cleaned_words = [word for word in words if word not in self.__stop_words]
        
        #returning cleaned text
        return " ".join(cleaned_words)
    
    def predict(self, text: str):
        #cleaning text
        cleaned_text = self.__clean_text(text)
        
        #vectorizing text
        vectorized_text = self.__vectorizor.transform([cleaned_text])
        
        #making prediction
        prediction = self.__model.predict(vectorized_text)
        
        #returning sentiment
        return self.__sentiment_mapping[prediction[0]]
    
if __name__ == "__main__":
    negative_reviews = [
        """Won’t buy again
I was over the moon when I got this. I love sunflowers and it looked exactly like they advertised, 
unfortunately I wore this piercing for one weekend. By Sunday night the sunflower had came off. I will 
not be purchasing from this company again. Very disappointed.""",

        """Don’t waste your time!

These are AWFUL. They are see through, the fabric feels like tablecloth, and they fit like children’s clothing.
Customer service did seem to be nice though, but I regret missing my return date for these. I wouldn’t even donate 
them because the quality is so poor.""",

        """Won't buy another one The LED ring works, but the remainder of the sign is too dark."""
    ]
    
    neutral_reviews = [
        """Just for looks

The presentation is great, but the flavor is ordinary, not impressed at all. I still would buy regular ones and decorate myself for less money.""",

        """Good for a backup and not expensive.

Not the best, but did help out a little for wind. They discolor fast and they are really big. I used electrical tape to keep it on.""",

        """Three Stars

Fits s little weird and pretty stiff but does it’s job"""
    ]
    
    positive_reviews = [
        """My 2-year old grandson loves this series!

My 2-year old grandson loves this series. Pictures are great; nice sturdy board book that is easy for him to use. 
He asks for this book to be read over and over.""",

        """Replacement Band

This replacement band is perfect. Great quality and durable. Excellent price and customer service!""",

        """Color is soooooooo pretty.

Shipped fast. Great product"""
    ]
    
    analyzer = SentimentAnalyser()
    
    print("Analyzing negative reviews: ")
    for i, review in enumerate(negative_reviews, start=1):
        print(f"review {i} sentiment: {analyzer.predict(review)}")
        
    print("\n\nAnalyzing neutral reviews: ")
    for i, review in enumerate(neutral_reviews, start=1):
        print(f"review {i} sentiment: {analyzer.predict(review)}")
        
    print("\n\nAnalyzing Positive reviews: ")
    for i, review in enumerate(positive_reviews, start=1):
        print(f"review {i} sentiment: {analyzer.predict(review)}")