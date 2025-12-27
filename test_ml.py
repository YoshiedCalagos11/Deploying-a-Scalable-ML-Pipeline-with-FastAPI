import numpy as np
import pandas as pd

from ml.model import train_model, compute_model_metrics
from ml.data import process_data


def test_train_model_returns_model():
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])

    model = train_model(X, y)

    assert model is not None


def test_compute_model_metrics_outputs():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1


def test_process_data_outputs():
    data = pd.DataFrame(
        {
            "age": [25, 32],
            "workclass": ["Private", "Self-emp-not-inc"],
            "education": ["Bachelors", "Masters"],
            "salary": [">50K", "<=50K"],
        }
    )

    X, y, encoder, lb = process_data(
        data,
        categorical_features=["workclass", "education"],
        label="salary",
        training=True,
    )

    assert X.shape[0] == 2
    assert len(y) == 2
    assert encoder is not None
    assert lb is not None
