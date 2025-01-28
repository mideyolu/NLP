## #NLP/pipeline.py

import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_load import load_data, preprocess_data


#Class Instance
class SentimentPipeline:

    def __init__(self, model_path = "model/sentiment_model.pkl" , vectorizer_path = "model/tdfvector.pkl"):
        """
          Initialize the SentimentPipeline with a model and vectorizer path
        """

        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.vectorizer= None
        self.model = None
