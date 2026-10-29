"""Проверки семантических признаков и временных профилей без загрузки Transformer."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from semantic_post_features import (
    N_CLUSTERS,
    SEMANTIC_COLUMNS,
    _catalog,
    attach_semantic_features,
    build_semantic_post_features,
    distances_from_embeddings,
)
from semantic_user_profile import PROFILE_COLUMN, add_user_post_similarity
from train_mlp_semantic import train_epochs, fit_final as fit_semantic
from train_mlp_personalized import choose_epochs, fit_final as fit_personalized


def test_distances_have_expected_shape_and_finite_values():
    rng = np.random.default_rng(42)
    embedded = rng.normal(size=(20, 8))
    distances = distances_from_embeddings(embedded)
    assert distances.shape == (20, N_CLUSTERS)
    assert np.isfinite(distances).all()
    assert (distances >= 0).all()


def test_post_catalog_does_not_need_target_and_sorts_by_id():
    posts = pd.DataFrame({"id": [3, 1, 2], "text": ["a", "b", "c"], "target": [1, 0, 1]})
    result = _catalog(posts)
    assert result.post_id.tolist() == [1, 2, 3]
    assert list(result.columns) == ["post_id", "text"]
    with pytest.raises(ValueError, match="уникальным"):
        _catalog(pd.DataFrame({"post_id": [1, 1], "text": ["a", "a"]}))


def test_semantic_cache_reuse_and_reject_changed_catalog(tmp_path, monkeypatch):
    calls = []

    class FakeTransformer:
        def __init__(self, model_name, device):
            calls.append((model_name, device))

        def encode(self, texts, **kwargs):
            return np.arange(len(texts) * 6, dtype=np.float32).reshape(len(texts), 6)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeTransformer)
    )
    posts = pd.DataFrame({"post_id": range(1, 21), "text": [f"post {i}" for i in range(20)]})
    cache = tmp_path / "vectors.npz"
    initial = build_semantic_post_features(posts, cache_path=cache)
    cached = build_semantic_post_features(posts, cache_path=cache)
    assert len(calls) == 1
    assert cache.exists()
    assert initial.shape == (len(posts), 1 + N_CLUSTERS)
    pd.testing.assert_frame_equal(initial, cached)
    changed = posts.copy()
    changed.loc[0, "text"] = "changed"
    with pytest.raises(ValueError, match="Несовместимый кэш"):
        build_semantic_post_features(changed, cache_path=cache)


def test_attach_semantic_features_keeps_order_and_rejects_unknown_post():
    semantic = pd.DataFrame({"post_id": [2, 1], **{name: [0.2, 0.1] for name in SEMANTIC_COLUMNS}})
    events = pd.DataFrame({"post_id": [1, 2, 1], "target": [0, 1, 1]}, index=[19, 10, 40])
    result = attach_semantic_features(events, semantic)
    assert result.post_id.tolist() == [1, 2, 1]
    assert result.target.tolist() == [0, 1, 1]
    np.testing.assert_allclose(result[SEMANTIC_COLUMNS[0]], [0.1, 0.2, 0.1])
    with pytest.raises(ValueError, match="Не для каждого"):
        attach_semantic_features(pd.DataFrame({"post_id": [3]}), semantic)


def _profile_fixture():
    vectors = pd.DataFrame({
        "post_id": [10, 11, 12],
        "vector": [np.array([1., 0.]), np.array([0., 1.]), np.array([-1., 0.])],
    })
    feed = pd.DataFrame({
        "user_id": [7, 7, 7, 7],
        "post_id": [10, 10, 11, 12],
        "action": ["like"] * 4,
        "timestamp": ["2021-11-01 10:00", "2021-11-01 10:01", "2021-11-01 10:02", "2021-11-01 10:05"],
    })
    return vectors, feed


def test_profile_past_excludes_same_time_and_future_likes():
    vectors, feed = _profile_fixture()
    events = pd.DataFrame({
        "user_id": [7] * 4,
        "post_id": [10, 10, 12, 12],
        "timestamp": ["2021-11-01 09:59", "2021-11-01 10:00", "2021-11-01 10:03", "2021-11-01 10:06"],
        "target": [0, 1, 0, 1],
    }, index=[10, 90, 20, 11])
    result = add_user_post_similarity(events, feed, vectors, mode="past")
    expected = [0, 0, -1 / np.sqrt(2), 0]
    np.testing.assert_allclose(result[PROFILE_COLUMN], expected, atol=1e-6)
    changed = events.copy()
    changed["target"] = 1 - changed["target"]
    other = add_user_post_similarity(changed, feed, vectors, mode="past")
    np.testing.assert_allclose(other[PROFILE_COLUMN], result[PROFILE_COLUMN])
    assert result.index.tolist() == events.index.tolist()


def test_fixed_profile_has_cutoff_and_does_not_see_future_like():
    vectors, feed = _profile_fixture()
    events = pd.DataFrame({
        "user_id": [7, 7], "post_id": [10, 12],
        "timestamp": ["2021-11-01 10:03", "2021-11-01 10:06"],
    })
    result = add_user_post_similarity(
        events, feed, vectors, mode="fixed", cutoff=pd.Timestamp("2021-11-01 10:02:00+00:00")
    )
    np.testing.assert_allclose(result[PROFILE_COLUMN], [1/np.sqrt(2), -1/np.sqrt(2)], atol=1e-6)
    with pytest.raises(ValueError, match="позже cutoff"):
        add_user_post_similarity(events, feed, vectors, mode="fixed", cutoff=pd.Timestamp("2021-11-01 10:03"))


def test_profile_rejects_bad_timestamp_and_cutoff():
    vectors, feed = _profile_fixture()
    events = pd.DataFrame({"user_id": [7], "post_id": [10], "timestamp": [pd.NaT]})
    with pytest.raises(ValueError, match="timestamp"):
        add_user_post_similarity(events, feed, vectors, mode="past")
    events.loc[0, "timestamp"] = "2021-11-02"
    with pytest.raises(ValueError, match="cutoff"):
        add_user_post_similarity(events, feed, vectors, mode="fixed", cutoff=pd.NaT)


@pytest.mark.parametrize("epochs,patience", [(0, 2), (3, 0)])
def test_semantic_early_stopping_requires_positive_parameters(epochs, patience):
    with pytest.raises(ValueError, match="положительными"):
        train_epochs(pd.DataFrame(), pd.DataFrame(), epochs, patience)
    with pytest.raises(ValueError, match="положительными"):
        choose_epochs(pd.DataFrame(), pd.DataFrame(), [], epochs, patience)


def test_final_training_requires_at_least_one_epoch():
    with pytest.raises(ValueError, match="положительным"):
        fit_semantic(pd.DataFrame(), 0)
    with pytest.raises(ValueError, match="положительным"):
        fit_personalized(pd.DataFrame(), [], 0)
