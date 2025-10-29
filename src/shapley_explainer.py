import numpy as np
from typing import Any, Callable, Iterable
from math import factorial
from itertools import chain, combinations
from copy import deepcopy

class ShapleyExplainer:
    """Shapley Values implementation from scratch.

    This class provides a simple implementation of Shapley values for model
    interpretability. It estimates the contribution of each feature to the
    model's prediction by computing Shapley values across all possible
    feature subsets.

    Attributes:
        model (Callable[[np.ndarray], float]):
            The predictive model to be explained. Must accept a NumPy array
            and return a prediction vector or scalar (e.g., `.predict`).
        background_dataset (np.ndarray):
            Background dataset used for estimating feature contributions.
            Typically a representative sample of the input data.
    """

    def __init__(self, model: Callable[[np.ndarray], np.ndarray], background_dataset: np.ndarray) -> None:
        """Initializes the Shapley explainer.

        Args:
            model (Callable[[np.ndarray], np.ndarray]):
                The model to explain (e.g., `estimator.predict`).
            background_dataset (np.ndarray):
                The dataset used as a background reference for computing Shapley values.
                Shape should be (n_background, n_features).
        """
        # TODO: Store `model` and `background_dataset` on `self`.
        self.model = model
        self.background_dataset = background_dataset

    def shapley_values(self, X: np.ndarray) -> np.ndarray:
        """Compute Shapley values for each instance and feature.

        Args:
            X (np.ndarray):
                Input samples for which Shapley values are computed (shape: n_instances x n_features).

        Returns:
            np.ndarray:
                A 2D array of Shapley values with the same shape as `X`.
        """
        # TODO:
        # 1) Create an array of shape equal to `X` to hold Shapley values.
        shapley_values = np.zeros_like(X)
        # 2) For each instance i and each feature j:
        #       - Compute single shapley value for feature j and instance i of X.
        #       - Store the result in the output array at [i, j].
        n_instances, n_features = X.shape
        for i in range(n_instances):
            for j in range(n_features):
                shapley_values[i, j] = self._compute_single_shapley_value(j, X[i])
        # 3) Return the filled Shapley array.
        return shapley_values

    def _compute_single_shapley_value(self, feature: int, instance: np.ndarray) -> float:
        """Compute the Shapley value for a single feature in one instance.

        Implements the Shapley formula (weighted average of marginal contributions)
        across all subsets that do not include the current feature.

        Args:
            feature (int):
                Index of the feature for which the Shapley value is computed.
            instance (np.ndarray):
                The input instance (1D, length = n_features).

        Returns:
            float:
                The Shapley value for the given feature.
        """
        # TODO:
        # 1) Determine the total number of features from `instance`.
        n_features = len(instance)
        # 2) Initialize an accumulator for the Shapley value.
        accumulator = 0.0
        # 3) Iterate over all subsets of "other" features (i.e., excluding `feature`):
        #       - Let S be such a subset (a tuple of indices).
        #       - Compute model expectation with S only: E[f(X) | X_S = instance_S]
        #         (use `_subset_model_approximation(S, instance)`).
        #       - Compute model expectation with S ∪ {feature}:
        #         `_subset_model_approximation(S_plus_feature, instance)`.
        #       - Compute the permutation weight using `_permutation_factor(n_features, len(S))`.
        #       - Add weight * (with_feature - without_feature) to the accumulator.
    
        for S in self._get_all_other_feature_subsets(n_features, feature):
            S_plus_feature = S + (feature, )
            without_feature = self._subset_model_approximation(S, instance)
            with_feature = self._subset_model_approximation(S_plus_feature, instance)
            weight = self._permutation_factor(n_features, len(S))
            accumulator += weight*(with_feature-without_feature)
            
        # 4) Return the accumulated value.
        # Hints:
        # - To form S ∪ {feature}, concatenate the tuple with `(feature,)`.
        # - Keep computations in float to avoid integer division issues.
        return float(accumulator)

    def _get_all_subsets(self, items: list[int]) -> Iterable[tuple[int, ...]]:
        """Generate all subsets of a list.

        Args:
            items (list[int]):
                List of feature indices.

        Returns:
            Iterable[tuple[int, ...]]:
                Iterator over all possible subsets (including empty and full).
        """
        # TODO:
        # 1) Return an iterator over all k-combinations for k = 0..len(items).
        # Hints:
        # - Use `itertools.combinations(items, r)` inside a generator expression.
        # - You can flatten with `itertools.chain.from_iterable(...)`.
        return chain.from_iterable(combinations(items, r) for r in range((len(items)+1)))

    def _get_all_other_feature_subsets(self, n_features: int, feature_of_interest: int) -> Iterable[tuple[int, ...]]:
        """Generate all subsets of features excluding the feature of interest.

        Args:
            n_features (int):
                Total number of features.
            feature_of_interest (int):
                Index of the feature to exclude.

        Returns:
            Iterable[tuple[int, ...]]:
                Iterator of feature index subsets not containing the feature of interest.
        """
        # TODO:
        # 1) Build a list of all feature indices except `feature_of_interest`.
        other_features = [i for i in range(n_features) if i!=feature_of_interest]
        # 2) Return all subsets of that list by calling `_get_all_subsets(...)`.
        # Hints:
        # - A list comprehension over range(n_features) works well for step (1).
        return self._get_all_subsets(other_features)

    def _permutation_factor(self, n_features: int, n_subset: int) -> float:
        """Compute the permutation weighting factor for a subset.

        This factor ensures fair averaging across feature subsets
        in the Shapley value computation.

        Args:
            n_features (int):
                Total number of features.
            n_subset (int):
                Number of features in the subset (|S|).

        Returns:
            float:
                Permutation weight for the subset:
                |S|! * (M - |S| - 1)! / M!  where M = n_features.
        """
        # TODO:
        # 1) Translate the formula into code using factorials.
        weight = factorial(n_subset)*factorial(n_features - n_subset -1)/factorial(n_features)
        # 2) Return the resulting float.
        # Hints:
        # - Use `math.factorial` imported above.
        return weight

    def _subset_model_approximation(self, feature_subset: tuple[int, ...], instance: np.ndarray) -> float:
        """Approximate the model output conditioned on a subset of features.

        This simulates E[f(X) | X_S = instance_S] by:
        - Copying the background dataset.
        - Overwriting the columns in S with the corresponding values from `instance`.
        - Predicting on the modified background.
        - Returning the mean prediction as the conditional expectation.

        Args:
            feature_subset (tuple[int, ...]):
                Indices of the features to condition on.
            instance (np.ndarray):
                Instance whose feature values are used to overwrite the background.

        Returns:
            float:
                The mean model output given the subset of features.
        """
        # TODO:
        # 1) Make a copy of the background dataset (avoid mutating the original).
        dataset = deepcopy(self.background_dataset)
        n_features = len(dataset)
        # 2) For each feature index j in 0..(n_features-1):
        #       - If j is in `feature_subset`, replace the j-th column of the copy
        #         with the scalar value `instance[j]` (broadcast to all rows).
        for j in range(0, n_features-1):
            if j in feature_subset:
                dataset[:, j] = instance[j]
        # 3) Call the model on the modified background to get predictions.
        predictions = self.model(dataset)
        # 4) Return the mean of those predictions as a scalar float.
        # Hints:
        # - Column assignment can broadcast a scalar to all rows in NumPy.
        # - Some models return shape (n,) and others (n, 1); take the mean robustly.
        return float(np.mean(predictions))