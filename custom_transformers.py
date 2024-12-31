import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class RareTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for replacing rare categories in a dataset with a specified replacement value.

    This transformer computes the relative frequency of each unique value across the dataset
    and replaces values below a specified threshold with a replacement value.

    Attributes:
        threshold (float): The minimum relative frequency for a value to be retained.
        replacement (str): The replacement value for rare categories.
        global_value_counts_ (dict): A dictionary containing the global frequency of each value.
        feature_names_in_ (list[str]): Names of the features in the input data.
    """

    def __init__(self, threshold: float = 0.05, replacement: str = "Rare"):
        """
        Initializes the RareTransformer.

        Args:
            threshold (float): The minimum relative frequency (proportion of dataset) for a value to be kept.
            replacement (str): The value to replace low-frequency terms with. Defaults to "Rare".
        """
        self.threshold = threshold
        self.replacement = replacement
        self.global_value_counts_: dict = {}
        self.feature_names_in_ = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray | None = None,
    ) -> "RareTransformer":
        """
        Fits the transformer by computing the global frequencies of values in the dataset.

        Args:
            X (pd.DataFrame | np.ndarray): The input categorical data.
            y (Optional[pd.DataFrame | np.ndarray]): Ignored. Included for compatibility with scikit-learn API.

        Returns:
            RareTransformer: The fitted instance of the transformer.

        Raises:
            ValueError: If the input data is not a pandas DataFrame or a NumPy array.
        """
        X = self._ensure_dataframe(X)
        self.feature_names_in_ = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        X = X.astype(str)
        total_count = X.size
        self.global_value_counts_ = X.stack().value_counts() / total_count
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """
        Transforms the data by replacing rare values with the replacement value.

        Args:
            X (pd.DataFrame | np.ndarray): The input categorical data.

        Returns:
            pd.DataFrame | np.ndarray: The transformed data.

        Raises:
            ValueError: If the input data is not a pandas DataFrame or a NumPy array.
        """
        X = self._ensure_dataframe(X)
        X = X.astype(str)
        X_transformed = X.applymap(self._replace_low_frequency)
        return X_transformed if isinstance(X, pd.DataFrame) else X_transformed.to_numpy()

    def _replace_low_frequency(self, value: str) -> str:
        """
        Replaces a value if its relative frequency is below the threshold.

        Args:
            value (str): The categorical value.

        Returns:
            str: The original value or the replacement value.
        """
        return value if self.global_value_counts_.get(value, 0) >= self.threshold else self.replacement

    def _ensure_dataframe(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """
        Ensures the input is a pandas DataFrame for processing.

        Args:
            X (pd.DataFrame | np.ndarray): The input data.

        Returns:
            pd.DataFrame: The data as a pandas DataFrame.

        Raises:
            ValueError: If the input is not a pandas DataFrame or NumPy array.
        """
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            raise ValueError("Input must be a pandas DataFrame or a NumPy array.")

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """
        Returns feature names after transformation.

        Args:
            input_features (Optional[List[str]]): Optional input feature names. If None, uses fitted feature names.

        Returns:
            List[str]: Feature names after transformation.

        Raises:
            ValueError: If the transformer has not been fitted yet.
        """
        if self.feature_names_in_ is None:
            raise ValueError("Transformer must be fitted before calling `get_feature_names_out`.")
        return input_features or self.feature_names_in_


class SimpleImputerWithNames(SimpleImputer):
    """
    An extension of SimpleImputer to retain column names in the output.

    Attributes:
        feature_names_in_ (List[str]): Names of the features in the input data.
    """

    def __init__(self, **kwargs):
        """
        Initializes the SimpleImputerWithNames.

        Args:
            **kwargs: Arguments to be passed to the underlying SimpleImputer.
        """
        super().__init__(**kwargs)
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> "SimpleImputerWithNames":
        """
        Fits the imputer and saves column names.

        Args:
            X (pd.DataFrame | np.ndarray): The input data to fit.
            y (Optional[np.ndarray]): Ignored. Included for compatibility with scikit-learn API.

        Returns:
            SimpleImputerWithNames: The fitted instance of the imputer.

        Raises:
            ValueError: If the input data is not a pandas DataFrame or a NumPy array.
        """
        self.feature_names_in_ = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        return super().fit(X, y)

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """
        Transforms the data and retains column names if the input is a DataFrame.

        Args:
            X (pd.DataFrame | np.ndarray): The input data to transform.

        Returns:
            pd.DataFrame | np.ndarray: The transformed data with column names retained (if applicable).

        Raises:
            ValueError: If the input data is not a pandas DataFrame or a NumPy array.
        """
        result = super().transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, columns=self.feature_names_in_, index=X.index)
        return result

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """
        Returns feature names after transformation.

        Args:
            input_features (Optional[List[str]]): Optional input feature names. If None, uses fitted feature names.

        Returns:
            List[str]: Feature names after transformation.

        Raises:
            ValueError: If the transformer has not been fitted yet.
        """
        if input_features is not None:
            return input_features
        if self.feature_names_in_ is None:
            raise ValueError("The transformer must be fitted before calling `get_feature_names_out`.")
        return self.feature_names_in_
