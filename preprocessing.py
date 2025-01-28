## #NLP/preprocessing.py
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

#Tokenization process
def process_text(text):
    """
      Function to proceess the text data

      Parameter:
        text : str : text data

        Returns:
            tokens : list : list of tokens
    """

    #separating the sentences into words
    tokens = word_tokenize(text)

    #removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    #Lemmatization
    lementizer = WordNetLemmatizer()
    lementizer_tokens = [lementizer.lemmatize(word) for word in filtered_tokens]

    return lementizer_tokens
