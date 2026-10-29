"""Проверки абляции и частотного кодирования без подключения к БД."""

import numpy as np
import pandas as pd
import pytest

import ablation_study
from city_encoder import CityFrequencyEncoder
from features import MODEL_FEATURES, SPLIT_DATE


def test_feature_sets_have_correct_sizes_and_subset_of_model_features():
    expected = {
        "history_26": 26,
        "history_cluster_27": 27,
        "history_length_28": 28,
        "history_text_29": 29,
        "full_32": 32,
    }
    assert set(ablation_study.FEATURE_SETS) == set(expected)
    for name, columns in ablation_study.FEATURE_SETS.items():
        assert len(columns) == expected[name]
        assert len(columns) == len(set(columns))
        assert set(columns).issubset(MODEL_FEATURES)


def test_new_ablation_feature_sets_change_only_text_features():
    base = ablation_study.FEATURE_SETS["history_26"]
    assert ablation_study.FEATURE_SETS["history_cluster_27"] == base + ["text_cluster"]
    assert ablation_study.FEATURE_SETS["history_length_28"] == base + ["text_length", "word_count"]


def test_validation_receives_no_events_after_test_boundary(monkeypatch):
    feed = pd.DataFrame({
        "timestamp": [
            "2021-11-01 00:00", "2021-11-29 00:00", "2021-12-12 00:00",
            "2021-12-13 00:00",
        ],
        "target": [0, 1, 0, 1],
    })
    users = pd.DataFrame({"user_id": [1]})
    posts = pd.DataFrame({"post_id": [2]})
    called = {}

    def fake_posts(frame, fit_pca):
        assert frame is posts
        assert fit_pca is False
        return "static_posts"

    def fake_temporal(frame_users, static_posts, development, split_date):
        assert frame_users is users
        assert static_posts == "static_posts"
        assert split_date == ablation_study.INNER_SPLIT_DATE
        called["development"] = development.copy()
        return pd.DataFrame({"target": [0, 1]}), pd.DataFrame({"target": [0, 1]}), {}

    monkeypatch.setattr(ablation_study, "build_post_features", fake_posts)
    monkeypatch.setattr(ablation_study, "build_temporal_datasets", fake_temporal)
    ablation_study.prepare_validation(users, posts, feed)
    development = called["development"]
    assert len(development) == 3
    assert development["timestamp"].max() <= SPLIT_DATE
    assert len(feed) == 4  # Исходная таблица не изменилась.


@pytest.mark.parametrize("max_epochs,patience", [(0, 3), (5, 0), (-1, 1)])
def test_invalid_mlp_early_stopping_settings_rejected(max_epochs, patience):
    with pytest.raises(ValueError, match="положительными"):
        ablation_study.evaluate_mlp(pd.DataFrame(), pd.DataFrame(), [], max_epochs, patience)


def test_city_frequency_uses_train_only_and_normalizes_missing_values():
    train = pd.DataFrame({"city": ["Moscow", "Moscow", None, "Kazan"]})
    future = pd.DataFrame({"city": ["Moscow", "NewCity", None, "Kazan"]})
    encoder = CityFrequencyEncoder().fit(train, y=np.array([0, 1, 1, 0]))
    result = encoder.transform(future).ravel()
    np.testing.assert_allclose(result, [0.5, 0.0, 0.25, 0.25])
    assert encoder.frequencies_["Moscow"] == 0.5
    assert "NewCity" not in encoder.frequencies_


def test_city_encoder_accepts_two_dimensional_numpy_and_checks_shape():
    train = np.array([["A"], ["B"], ["A"]], dtype=object)
    encoder = CityFrequencyEncoder().fit(train)
    np.testing.assert_allclose(
        encoder.transform(np.array([["A"], ["C"]], dtype=object)).ravel(),
        [2 / 3, 0],
    )
    with pytest.raises(ValueError, match="один столбец"):
        encoder.transform(np.array([["A", "B"]], dtype=object))
