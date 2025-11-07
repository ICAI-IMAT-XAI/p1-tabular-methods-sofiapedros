# test_shap_explainer.py
"""
Tests for ShapExplainer.

These tests:
1) Compare ShapExplainer values to shap.KernelExplainer on a few instances.
2) Verify SHAP additivity: sum(phi_j) ~= f(x) - E[f(X_background)].

Run with:
    pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest
import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

import shap

# Adjust this import to match where you placed the class
# e.g., from mypackage.explainers import ShapExplainer
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shapley_explainer import ShapleyExplainer  # <-- update if your module name differs


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


@pytest.fixture(scope="module")
def data():
    """Load and split the diabetes dataset."""
    X, y = load_diabetes(as_frame=False, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def models(data):
    """Train or load models from disk."""
    X_train, _, y_train, _ = data
    model_paths = {
        "linear": os.path.join(MODELS_DIR, "linear_model.joblib"),
        "random_forest": os.path.join(MODELS_DIR, "random_forest.joblib"),
        "gradient_boosting": os.path.join(MODELS_DIR, "gradient_boosting.joblib"),
    }

    models = {}

    # Try to load models, otherwise train and save
    for key, path in model_paths.items():
        if os.path.exists(path):
            print(f"🔹 Loading cached model: {key}")
            models[key] = joblib.load(path)
        else:
            print(f"⚙️ Training new model: {key}")
            if key == "linear":
                model = LinearRegression().fit(X_train, y_train)
            elif key == "random_forest":
                model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
            elif key == "gradient_boosting":
                model = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)
            else:
                raise ValueError(f"Unknown model key: {key}")

            joblib.dump(model, path)
            models[key] = model

    return models


@pytest.fixture(scope="module")
def background_and_instances(data):
    """Provide a small background and test subset for SHAP comparison."""
    X_train, X_test, _, _ = data
    rng = np.random.default_rng(42)
    bg_size = min(50, X_train.shape[0])
    background_idx = rng.choice(np.arange(X_train.shape[0]), size=bg_size, replace=False)
    X_background = X_train[background_idx]
    X_instances = X_test[:3]
    return X_background, X_instances


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def kernel_shap_values(model, X_background: np.ndarray, X_instances: np.ndarray) -> np.ndarray:
    """Compute exact Kernel SHAP values (2^features) for small problems."""
    n_features = X_instances.shape[1]
    explainer = shap.KernelExplainer(model.predict, X_background)
    values = explainer.shap_values(X_instances, nsamples=2 ** n_features)
    return np.asarray(values)


def ours_shap_values(model, X_background: np.ndarray, X_instances: np.ndarray) -> np.ndarray:
    """Compute SHAP values using ShapleyExplainer."""
    explainer = ShapleyExplainer(model.predict, X_background)
    return explainer.shapley_values(X_instances)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("model_key", ["linear", "random_forest", "gradient_boosting"])
def test_matches_kernel_shap(models, background_and_instances, model_key):
    """ShapExplainer should approximately match shap.KernelExplainer."""
    model = models[model_key]
    X_background, X_instances = background_and_instances

    lib_vals = kernel_shap_values(model, X_background, X_instances)
    our_vals = ours_shap_values(model, X_background, X_instances)

    assert lib_vals.shape == our_vals.shape == X_instances.shape
    assert np.allclose(our_vals, lib_vals, rtol=1e-3, atol=5e-2), (
        f"Disagreement with Kernel SHAP for {model_key}. "
        f"Max abs diff: {np.max(np.abs(our_vals - lib_vals)):.5f}"
    )


@pytest.mark.parametrize("model_key", ["linear", "random_forest", "gradient_boosting"])
def test_additivity_property(models, background_and_instances, model_key):
    """Sum of SHAP values equals f(x) - E[f(X_background)]."""
    model = models[model_key]
    X_background, X_instances = background_and_instances

    our_vals = ours_shap_values(model, X_background, X_instances)
    f_bg = np.mean(model.predict(X_background))
    lhs = our_vals.sum(axis=1)
    rhs = model.predict(X_instances) - f_bg

    assert np.allclose(lhs, rhs, rtol=1e-5, atol=1e-4), (
        f"Additivity failed for {model_key}. "
        f"Max abs diff: {np.max(np.abs(lhs - rhs)):.6f}"
    )