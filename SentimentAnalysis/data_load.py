## #NLP/data_load.py

import pandas as pd
from preprocessing import process_text


def load_data(file_path):
    """
      Function to load the data from the file

      Parameter:
        file_path : str : path of the file

        Returns:
            data : DataFrame : dataframe of the data
    """
    data = pd.read_csv(file_path, sep=";", names=["text", "label"])

    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("Columns not found in the data")

    return data


def preprocess_data(data):
    """
      Function to preprocess the data

      Parameter:
        data : DataFrame : dataframe of the data

        Returns:
            data : DataFrame : dataframe of the data
    """

    # Check if text and label columns are present
    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("Columns not found in the data")

    data["cleaned_text"]= data["text"].apply(process_text)

    return data[["cleaned_text", "label"]]


def save_processed_data(data, output_file):
    """
      Function to save the processed data

        Parameter:
            data : DataFrame : dataframe of the data
            output_path : str : path to save the data

        Returns:
            None
    """
    data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")


# #Example Usage
# if __name__ == "__main__":
#     file_path = "data/raw/train.txt"
#     output_path= "data/processed/train.txt"


#     try:

#         #load the data
#         data = load_data(file_path=file_path)

#         #preprocess the data
#         processed_data = preprocess_data(data=data)

#         print(f"First 5 rows of processed data.\n{preprocess_data}")

#         #saving the processed data

#         save_processed_data(data=processed_data, output_file=output_path)

#     except Exception as e:
#         print(f"Error: {e}")
