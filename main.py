import os
import time

import numpy as np
import pandas as pd
from icecream import argumentToString, ic
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import custom_transformers


def ic_time_formatter():
    """
    Formats and returns the time elapsed since the last call to this function.

    Returns:
        str: A formatted string representing the elapsed time in seconds.
    """
    global global_ic_time_formatter_last_start_time

    try:
        timing = time.time() - global_ic_time_formatter_last_start_time
    except NameError:
        timing = 0

    global_ic_time_formatter_last_start_time = time.time()
    return f"{timing:.3f}s | "


@argumentToString.register(np.ndarray)
def _(obj):
    """
    Custom formatter for NumPy arrays in icecream debug output.

    Args:
        obj (np.ndarray): The NumPy array to format.

    Returns:
        str: A string representation of the array's shape and data type.
    """
    return f"ndarray, shape={obj.shape}, dtype={obj.dtype}"


@argumentToString.register(pd.DataFrame)
def _(obj):
    """
    Custom formatter for pandas DataFrames in icecream debug output.

    Args:
        obj (pd.DataFrame): The DataFrame to format.

    Returns:
        str: A string representation of the DataFrame's shape.
    """
    return f"DataFrame, shape={obj.shape}"


def load_data(dir_path: str, file_name: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        dir_path (str): The directory path containing the file.
        file_name (str): The name of the file to load.

    Returns:
        pd.DataFrame: The loaded data.
    """
    file_path = os.path.join(dir_path, file_name)
    return pd.read_csv(file_path)


def get_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list, list, str]:
    """
    Extracts features and target columns from the data.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): The feature columns.
            - y (pd.DataFrame): The target column.
            - cat_features (list): List of categorical feature names.
            - num_features (list): List of numerical feature names.
            - target (str): The target column name.
    """
    cat_features = ["PropType"]
    num_features = ["Stories", "Year_Built", "Units", "FinishedSqft"]
    features = cat_features + num_features
    target = "Sale_price"

    X = data[features]
    y = data[target]

    return X, y, cat_features, num_features, target


def build_preprocessing_pipeline(
    X: pd.DataFrame, y: pd.DataFrame, cat_features: list, num_features: list, target: str
) -> ColumnTransformer:
    """
    Builds a preprocessing pipeline for both categorical and numerical features.

    Args:
        X (pd.DataFrame): The feature data.
        y (pd.DataFrame): The target data.
        cat_features (list): List of categorical feature names.
        num_features (list): List of numerical feature names.
        target (str): The target column name.

    Returns:
        ColumnTransformer: A transformer object for preprocessing features.
    """
    numeric_preprocessor = Pipeline(
        [
            ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocessor = Pipeline(
        [
            (
                "imputation_constant",
                SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="missing"),
            ),
            ("rare", custom_transformers.RareTransformer(threshold=0.05)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("categorical", categorical_preprocessor, cat_features),
            ("numerical", numeric_preprocessor, num_features),
        ]
    )
    return preprocessor


def main():
    """
    Main function to execute the preprocessing pipeline on input data.

    Loads the data, extracts features, builds a preprocessing pipeline, and applies it.
    Outputs the results using the icecream debug tool.
    """
    ic.configureOutput(prefix=ic_time_formatter)

    ic("\n------Start Main------\n")

    DATA_PATH = "data"
    FILE_NAME = "armslengthsales_2024_valid.csv"

    data = ic(load_data(DATA_PATH, FILE_NAME))

    X, y, CAT_FEATURES, NUM_FEATURES, TARGET = get_features(data)

    pipeline = ic(build_preprocessing_pipeline(X, y, CAT_FEATURES, NUM_FEATURES, TARGET))

    ic(pipeline.fit_transform(X, y))


if __name__ == "__main__":
    main()
