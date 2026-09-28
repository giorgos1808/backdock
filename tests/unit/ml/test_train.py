import importlib.util
import pathlib
import sys
import pandas as pd
import pytest

COMPONENT = pathlib.Path(__file__).resolve().parents[3] / "ml" / "components" / "train" / "train.py"


def load_component():
    spec = importlib.util.spec_from_file_location("train_component", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_component"] = module
    spec.loader.exec_module(module)
    return module


train = load_component()


def synthetic_history(rows=600, seed=0):
    """Quantities driven by their own lag plus noise, which is the structure
    the real features have.
    """
    rng = pd.Series(range(rows))
    frame = pd.DataFrame(
        {
            "lag_1": 10 + (rng % 17) * 0.7,
            "lag_2": 9 + (rng % 13) * 0.6,
            "rolling_avg_3": 10 + (rng % 11) * 0.5,
            "month": (rng % 12) + 1,
            "day_of_week": rng % 7,
            "supplier_code": rng % 6,
        }
    )
    frame["quantity"] = frame["lag_1"] * 0.8 + frame["rolling_avg_3"] * 0.2 + (rng % 5) * 0.3
    return frame


@pytest.fixture(scope="module")
def fitted():
    data = synthetic_history()
    return train.train_model(data.iloc[:450], data.iloc[450:])


class TestTrainingConfiguration:
    def test_optimises_the_metric_it_is_judged_on(self, fitted):
        """evaluate.py's gate is MAPE, but LightGBM defaults to squared error.
        Training on L2 while being judged on MAPE cost about 2.5pp and left the
        model losing to the baseline on 9 of 10 splits.
        """
        assert fitted.get_params()["objective"] == "mape"


    def test_early_stopping_actually_consumes_the_validation_split(self, fitted):
        """The validation split was always built and passed in, but without a
        callback nothing used it and all n_estimators trees were built
        regardless — train 17.9% against test 25.4%, textbook overfitting.

        best_iteration_ is only set when early stopping runs, and stopping
        short of n_estimators is what proves the validation scores were
        actually consulted.
        """
        assert fitted.best_iteration_ is not None
        assert fitted.best_iteration_ < fitted.get_params()["n_estimators"]
        assert fitted.booster_.num_trees() == fitted.best_iteration_


    def test_validation_is_scored_on_the_gate_metric(self, fitted):
        """Early stopping has to stop on MAPE. Stopping on the default L2 would
        reintroduce the objective mismatch through the back door.
        """
        assert "mape" in fitted.best_score_["valid_0"]


    def test_leaves_are_not_allowed_to_form_from_a_single_row(self, fitted):
        """min_child_samples=1 let the model memorise individual rows."""
        assert fitted.get_params()["min_child_samples"] >= 20


    def test_uses_the_features_data_prep_produces(self):
        assert train.FEATURE_COLUMNS == [
            "lag_1",
            "lag_2",
            "rolling_avg_3",
            "month",
            "day_of_week",
            "supplier_code",
        ]
        assert train.TARGET_COLUMN == "quantity"
        assert train.CATEGORICAL_COLUMNS == ["supplier_code"]


class TestTrainedModel:
    def test_predicts_one_value_per_row(self, fitted):
        data = synthetic_history(rows=40, seed=1)

        predictions = fitted.predict(data[train.FEATURE_COLUMNS])

        assert len(predictions) == len(data)


    def test_learns_something_from_the_signal(self, fitted):
        """A sanity floor, not an accuracy claim: on data where quantity is a
        deterministic function of the lags, the fit should beat predicting the
        training mean.
        """
        data = synthetic_history(rows=200, seed=2)
        actual = data["quantity"]
        predicted = fitted.predict(data[train.FEATURE_COLUMNS])

        model_error = (actual - predicted).abs().mean()
        mean_error = (actual - actual.mean()).abs().mean()

        assert model_error < mean_error
