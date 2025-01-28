import os
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_load import load_data, preprocess_data, save_processed_data
from preprocessing import process_text

class SentimentPipeline:

    def __init__(self, model_path="models/sentiment_model.pkl", vectorizer_path="models/tfidf_vectorizer.pkl"):
        """
        Initialize the SentimentPipeline with model and vectorizer paths.
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None

    def load_model_and_vectorizer(self):
        """
        Load the model and vectorizer from their respective paths.
        """
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            print(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            print(f"Loading vectorizer from {self.vectorizer_path}")
            self.vectorizer = joblib.load(self.vectorizer_path)
        else:
            raise FileNotFoundError("Model or Vectorizer file not found. Train the model first.")

    def save_model_and_vectorizer(self, pipeline):
        """
        Save the trained model and vectorizer to their respective paths.
        """
        print("Saving the model and vectorizer...")
        self.model = pipeline.named_steps["classifier"]
        self.vectorizer = pipeline.named_steps["vectorizer"]
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        print(f"Model and vectorizer saved to {self.model_path} and {self.vectorizer_path}")

    def pipeline(self, train_file):
        """
        Train the pipeline:
        - Loads and preprocesses data
        - Trains the model
        - Saves the model and vectorizer
        """
        try:
            self.load_model_and_vectorizer()
        except FileNotFoundError:
            print("Training a new model...")
            data = load_data(train_file)
            processed_data = preprocess_data(data)

            X, y = processed_data["cleaned_text"], processed_data["label"]

            pipeline = Pipeline([
                ("vectorizer", TfidfVectorizer(max_features=4000)),
                ("classifier", LGBMClassifier(verbose=0))
            ])
            pipeline.fit(X, y)
            self.save_model_and_vectorizer(pipeline)

    def evaluate(self, test_file, name):
        """
        Evaluate the model on a test dataset.
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer not loaded. Train or load the pipeline first.")

        print("Loading and preprocessing test data...")
        test_data = load_data(test_file)
        processed_test_data = preprocess_data(test_data)

        X_test = processed_test_data["cleaned_text"]
        y_test = processed_test_data["label"]

        X_test_transformed = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_transformed)

        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"{name} Evaluation")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(conf_matrix)

        processed_output_path = test_file.replace("raw", "processed")
        os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)
        save_processed_data(processed_test_data, processed_output_path)

        return {
            "classification_report": report,
            "confusion_matrix": conf_matrix
        }

    def predict(self, text):
        """
        Predict sentiment for a given text input.
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer not loaded. Train or load the pipeline first.")

        processed_text = " ".join(process_text(text))
        labels = ["joy", "sadness", "anger", "fear", "love", "surprise"]

        transformed_text = self.vectorizer.transform([processed_text])
        prediction_class = self.model.predict(transformed_text)

        if isinstance(prediction_class[0], str):
            prediction = prediction_class
        else:
            prediction = [labels[pred] for pred in prediction_class]

        return prediction


if __name__ == "__main__":
    train_file_path = "data/raw/train.txt"
    test_file_path = "data/raw/test.txt"

    sentiment_pipeline = SentimentPipeline()
    sentiment_pipeline.pipeline(train_file_path)

    test_results = sentiment_pipeline.evaluate(test_file_path, name="Test")
    sample_text = "I cant walk shop anywhere feel uncomfortable"
    prediction = sentiment_pipeline.predict(sample_text)
    print(f"Predicted Sentiment: {prediction}")
