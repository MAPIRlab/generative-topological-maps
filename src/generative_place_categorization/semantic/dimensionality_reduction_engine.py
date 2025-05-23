from sklearn.decomposition import PCA
from umap import UMAP
import numpy as np


class DimensionalityReductionEngine:
    """
    Engine for performing dimensionality reduction on feature matrices.
    """

    def reduce(self, feature_matrix: np.ndarray, target_dimension: int, method: str) -> np.ndarray:
        """
        Reduce the dimensionality of the given feature matrix.

        Args:
            feature_matrix (np.ndarray): The input feature matrix.
            target_dimension (int): The target number of dimensions.
            method (str): The dimensionality reduction method ('pca' or 'umap').

        Returns:
            np.ndarray: The reduced feature matrix.

        Raises:
            ValueError: If the feature matrix has fewer dimensions than the target dimension.
            NotImplementedError: If the specified method is not implemented.
        """
        if feature_matrix.shape[1] <= target_dimension:
            raise ValueError(
                f"Feature matrix size ({feature_matrix.shape[1]}) is lower than target dimension {target_dimension}"
            )

        if method == "pca":
            reducer = PCA(n_components=target_dimension)
        elif method == "umap":
            reducer = UMAP(n_components=target_dimension, random_state=42)
        else:
            raise NotImplementedError(
                f"Dimensionality reduction method {method} is not implemented.")

        return reducer.fit_transform(feature_matrix)
