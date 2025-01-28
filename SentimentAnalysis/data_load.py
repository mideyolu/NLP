import pandas as pd
from preprocessing import process_text

def load_data(file_path):
    """
    Load data from a file.
    """
    data = pd.read_csv(file_path, sep=";", names=["text", "label"])
    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("Columns 'text' and 'label' not found in the data.")
    return data

def preprocess_data(data):
    """
    Preprocess data by cleaning text and converting to string format.
    """
    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("Columns 'text' and 'label' not found in the data.")

    data["cleaned_text"] = data["text"].apply(lambda x: " ".join(process_text(x)))
    return data[["cleaned_text", "label"]]


def save_processed_data(data, output_file):
    """
    Save processed data to a file.
    """
    data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
