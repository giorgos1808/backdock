import sys
import pytest
import app.services.model_registry as model_registry
from app.services.model_registry import ForecastModel

PREDICT_ARGS = {
    "lag_1": 10.0,
    "lag_2": 8.0,
    "rolling_avg_3": 9.0,
    "month": 6,
    "day_of_week": 2,
    "supplier_name": "ACME Foods Ltd",
}


@pytest.fixture
def unwarned(monkeypatch):
    """The 'dependencies missing' warning is once-per-process, so reset the
    module flag rather than letting test order decide whether it fires.
    """
    monkeypatch.setattr(model_registry, "_warned_missing_dependencies", False)


@pytest.fixture
def scoring_stack_missing(monkeypatch, unwarned):
    monkeypatch.setitem(sys.modules, "azure.ai.ml", None)


class TestConfigured:
    @pytest.mark.parametrize(
        "args",
        [
            (None, None, None),
            ("sub", None, None),
            ("sub", "rg", None),
            ("sub", "rg", ""),
        ],
    )


    def test_incomplete_workspace_settings_are_not_configured(self, args):
        assert ForecastModel(*args).configured is False


    def test_all_three_present_is_configured(self):
        assert ForecastModel("sub", "rg", "workspace").configured is True


class TestRefresh:
    def test_unconfigured_returns_false_without_importing_anything(self, scoring_stack_missing):
        """The cheap path: no ML workspace means no model, decided before the
        import is even attempted. Poisoning sys.modules here proves it never
        gets that far.
        """
        assert ForecastModel(None, None, None).refresh() is False


    def test_missing_scoring_stack_falls_back_instead_of_raising(self, scoring_stack_missing):
        """The web image's normal state when enable_ml is on: AZURE_ML_* is
        set, but azure-ai-ml/lightgbm aren't installed.
        """
        model = ForecastModel("sub", "rg", "workspace")

        assert model.configured is True
        assert model.refresh() is False
        assert model.version is None


    def test_missing_scoring_stack_warns_once_only(self, scoring_stack_missing, caplog):
        """generate_forecast() calls refresh() on every forecast request, so a
        per-call warning would flood the logs.
        """
        model = ForecastModel("sub", "rg", "workspace")

        with caplog.at_level("WARNING", logger="app.services.model_registry"):
            model.refresh()
            model.refresh()
            model.refresh()

        warnings = [r for r in caplog.records if "scoring stack is not installed" in r.message]
        assert len(warnings) == 1


class TestPredict:
    def test_returns_none_when_no_model_is_loaded(self):
        """What makes the fallback work: forecasting.py treats None as 'no ML
        forecast available' and uses the baseline instead.
        """
        assert ForecastModel("sub", "rg", "workspace").predict(**PREDICT_ARGS) is None


    def test_returns_none_after_a_failed_refresh(self, scoring_stack_missing):
        model = ForecastModel("sub", "rg", "workspace")
        model.refresh()

        assert model.predict(**PREDICT_ARGS) is None


class TestMatchKey:
    """The shared rapidfuzz processor. Lives in app/services/text.py because
    document_intelligence, reconciliation and products all need the same
    normalisation.
    """

    def test_folds_accents_so_greek_capitals_compare_equal(self):
        from app.services.text import match_key

        assert match_key("Γάλα πλήρες") == match_key("ΓΑΛΑ ΠΛΗΡΕΣ")


    def test_normalises_final_sigma(self):
        """Capital Σ lowercases to σ in every position, but lower-case Greek
        writes ς at the end of a word — so "πλήρες" and "ΠΛΗΡΕΣ" would
        otherwise stay one character apart after everything else matched.
        """
        from app.services.text import match_key

        assert match_key("λήξης") == match_key("ΛΗΞΗΣ")


    def test_still_lowercases_and_replaces_punctuation(self):
        """rapidfuzz's own default_process behaviour is kept underneath: case
        folded, punctuation turned into spaces. It does not collapse the
        resulting runs of spaces, which is why this is not simply equality
        against the unpunctuated form.
        """
        from app.services.text import match_key

        assert match_key("TOMATO SAUCE 400G") == "tomato sauce 400g"
        assert match_key("Tomato-Sauce") == "tomato sauce"


    def test_fold_accents_preserves_length(self):
        """document_intelligence slices lot numbers out of the original string
        using spans found in the folded one, so the two must stay aligned.
        """
        from app.services.text import fold_accents

        for text in ["ΠΑΡΤΙΔΑ: Aé12", "ανάλωση κατά προτίμηση", "plain ascii", ""]:
            assert len(fold_accents(text)) == len(text)


    def test_handles_none_safely(self):
        from app.services.text import match_key

        assert match_key(None) == ""


class TestLocalModelPath:
    """LOCAL_FORECAST_MODEL_PATH is a development affordance: score with a
    model on disk instead of one fetched from the Azure ML registry, so the
    scoring path can be exercised without a workspace.
    """

    def test_a_local_path_alone_counts_as_configured(self):
        """Without it, refresh() returns before doing anything, because
        `configured` demanded the three AZURE_ML_* values.
        """
        model = ForecastModel(None, None, None, local_model_path="/some/dir")

        assert model.configured is True


    def test_still_unconfigured_with_neither(self):
        assert ForecastModel(None, None, None).configured is False


    def test_missing_files_fall_back_rather_than_raising(self, tmp_path, unwarned):
        """A path that holds no model is a misconfiguration, not a crash — the
        caller treats it as "no model available" like any other.
        """
        model = ForecastModel(None, None, None, local_model_path=str(tmp_path))

        assert model.refresh() is False
        assert model.predict(**PREDICT_ARGS) is None


    def test_local_path_wins_over_the_registry(self, tmp_path, unwarned):
        """Setting it is an explicit instruction, so it short-circuits the
        registry rather than being a fallback for it. An empty directory makes
        refresh fail *locally* — if the registry had been consulted instead
        this would have tried to reach Azure.
        """
        model = ForecastModel("sub", "rg", "workspace", local_model_path=str(tmp_path))

        assert model.refresh() is False


    def test_a_real_model_loads_and_is_versioned_local(self, tmp_path):
        """Recorded as `ml:local`, never as a registered version number, so a
        development model cannot be mistaken in the database or the UI for one
        that passed the registration gate.
        """
        lgb = pytest.importorskip("lightgbm")
        import numpy as np

        rows = 200
        features = np.column_stack(
            [
                np.arange(rows) % 17,
                np.arange(rows) % 13,
                np.arange(rows) % 11,
                np.arange(rows) % 12 + 1,
                np.arange(rows) % 7,
                np.arange(rows) % 6,
            ]
        ).astype(float)
        target = features[:, 0] * 2 + 1
        booster = lgb.train(
            {"objective": "regression", "verbose": -1, "min_data_in_leaf": 1},
            lgb.Dataset(features, label=target),
            num_boost_round=5,
        )
        booster.save_model(str(tmp_path / "model.txt"))
        (tmp_path / "supplier_categories.json").write_text('{"ACME Foods Ltd": 0}', encoding="utf-8")

        model = ForecastModel(None, None, None, local_model_path=str(tmp_path))

        assert model.refresh() is True
        assert model.version == "local"
        assert isinstance(model.predict(**PREDICT_ARGS), float)


    def test_a_loaded_local_model_is_not_reloaded_every_call(self, tmp_path):
        """generate_forecast() calls refresh() per forecast; re-reading the
        booster off disk each time would make a batch run needlessly slow.
        """
        lgb = pytest.importorskip("lightgbm")
        import numpy as np

        features = np.column_stack([np.arange(50) % 7] * 6).astype(float)
        booster = lgb.train(
            {"objective": "regression", "verbose": -1, "min_data_in_leaf": 1},
            lgb.Dataset(features, label=features[:, 0]),
            num_boost_round=3,
        )
        booster.save_model(str(tmp_path / "model.txt"))
        (tmp_path / "supplier_categories.json").write_text("{}", encoding="utf-8")

        model = ForecastModel(None, None, None, local_model_path=str(tmp_path))
        assert model.refresh() is True

        (tmp_path / "model.txt").unlink()

        assert model.refresh() is True
