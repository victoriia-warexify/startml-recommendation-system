"""Проверки FastAPI без PostgreSQL и файлов обученных моделей."""

import json
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as service
from features import (
    FEATURE_DEFINITION,
    MODEL_FEATURES,
    SPLIT_DATE,
    build_dataset_from_features,
    build_history_snapshot,
    snapshot_export_tables,
)


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.feature_names_in_ = np.array(MODEL_FEATURES)
        self.feature_definition_ = FEATURE_DEFINITION
        self.history_cutoff_ = SPLIT_DATE.isoformat()
        self.calls = []

    def predict_proba(self, X):
        self.calls.append(X.copy())
        p = X["text_length"].to_numpy(dtype=float) / 100
        return np.column_stack([1 - p, p])


@pytest.fixture
def api_context(monkeypatch):
    users = pd.DataFrame([
        {"user_id": uid, "gender": 0, "age": 27, "country": "RU",
         "city": "Kazan", "exp_group": 1, "os": "iOS", "source": "ads"}
        for uid in (200, 220, 401)
    ])
    posts = pd.DataFrame([
        {"post_id": 10, "topic": "tech", "text_cluster": "tech:c00",
         "text_length": 10, "word_count": 2, "text": "text 10"},
        {"post_id": 11, "topic": "tech", "text_cluster": "tech:c00",
         "text_length": 20, "word_count": 3, "text": "text 11"},
        {"post_id": 12, "topic": "food", "text_cluster": "food:c00",
         "text_length": 30, "word_count": 4, "text": "text 12"},
        {"post_id": 13, "topic": "food", "text_cluster": "food:c00",
         "text_length": 40, "word_count": 5, "text": "text 13"},
    ])
    feed = pd.DataFrame([
        {"user_id": 200, "post_id": 10, "timestamp": "2021-12-11 10:00",
         "action": "view", "target": 0},
        {"user_id": 200, "post_id": 10, "timestamp": "2021-12-11 10:05",
         "action": "like", "target": None},
        {"user_id": 220, "post_id": 12, "timestamp": "2021-12-11 11:00",
         "action": "view", "target": 1},
    ])
    snapshot = build_history_snapshot(users, posts.drop(columns="text"), feed)
    tables = snapshot_export_tables(snapshot)
    tables["posts_info"] = posts[["post_id", "text", "topic"]].copy()
    models = {"control": FakeModel("control"), "test": FakeModel("test")}

    monkeypatch.setattr(service, "load_model", lambda name: models[name])
    monkeypatch.setattr(service, "feature_table", lambda name: name)

    def fake_load_sql(sql, dtypes=None):
        for name in tables:
            if name == "posts_info":
                if "FROM public.post_text_df" in sql:
                    return tables[name].copy()
            elif f'FROM public."{name}"' in sql:
                return tables[name].copy()
        raise AssertionError(f"Неожиданный SQL: {sql}")

    monkeypatch.setattr(service, "load_sql", fake_load_sql)
    with TestClient(service.app) as client:
        yield client, models, snapshot, posts


def test_control_and_test_route_and_exclude_seen(api_context):
    client, models, _, _ = api_context
    dt = "2021-12-20T12:00:00"
    for user_id, expected_group, excluded in (
        (200, "control", 10),
        (220, "test", 12),
    ):
        response = client.get("/post/recommendations/", params={
            "user_id": user_id, "dt": dt, "limit": 3,
        })
        assert response.status_code == 200
        result = response.json()
        assert result["exp_group"] == expected_group
        ids = [post["id"] for post in result["recommendations"]]
        assert len(ids) == len(set(ids)) == 3
        assert excluded not in ids
        assert len(models[expected_group].calls) == 1


def test_api_features_match_frozen_offline_features(api_context):
    client, models, snapshot, posts = api_context
    dt = pd.Timestamp("2021-12-20T12:00:00")
    response = client.get("/post/recommendations/", params={
        "user_id": 200, "dt": dt.isoformat(), "limit": 3,
    })
    assert response.status_code == 200
    actual = models["control"].calls[-1].reset_index(drop=True)
    candidate_ids = posts.loc[posts.post_id != 10, "post_id"].tolist()
    rows = pd.DataFrame([
        {"user_id": 200, "post_id": post_id, "timestamp": dt,
         "action": "view", "target": 0}
        for post_id in candidate_ids
    ])
    expected = build_dataset_from_features(
        rows, snapshot["user_features"], snapshot["post_features"],
        snapshot["user_topic_features"], snapshot["user_cluster_features"],
    )[MODEL_FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_api_time_validation_and_unknown_user(api_context):
    client, _, _, _ = api_context
    url = "/post/recommendations/"
    assert client.get(url, params={"user_id": 9999, "dt": "2021-12-20"}).status_code == 404
    assert client.get(url, params={"user_id": 200, "dt": "2021-12-12"}).status_code == 422
    assert client.get(url, params={"user_id": 200, "dt": "2021-12-20", "limit": 0}).status_code == 422
    assert client.get(url, params={"user_id": 200, "dt": "2021-12-20", "limit": 101}).status_code == 422


def test_seen_history_is_strictly_before_request_time(api_context):
    _, _, _, _ = api_context
    saved = service.user_seen_by_id.copy()
    saved.loc[200, "seen_history_json"] = json.dumps([
        [10, "2021-12-11T10:00:00"],
        [11, "2021-12-20T12:00:00"],
    ])
    service.user_seen_by_id = saved
    assert service.get_seen_post_ids(200, datetime(2021, 12, 20, 12, 0)) == {10}
    assert service.get_seen_post_ids(200, datetime(2021, 12, 20, 12, 0, 1)) == {10, 11}


def test_broken_seen_json_is_not_silently_ignored(api_context):
    service.user_seen_by_id.loc[200, "seen_history_json"] = "{not valid json"
    with pytest.raises(ValueError, match="Некорректная история"):
        service.get_seen_post_ids(200, datetime(2021, 12, 20))


def test_empty_history_for_cold_start_user(api_context):
    client, _, _, _ = api_context
    assert service.get_seen_post_ids(401, datetime(2021, 12, 20)) == set()
    response = client.get("/post/recommendations/", params={
        "user_id": 401, "dt": "2021-12-20", "limit": 4,
    })
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 4


def test_groups_are_deterministic():
    for user_id in (1, 200, 220, 9999):
        assert service.get_exp_group(user_id) == service.get_exp_group(user_id)
    assert service.get_exp_group(200) == "control"
    assert service.get_exp_group(220) == "test"


def test_bad_probabilities_and_limit_are_rejected():
    class BadModel(FakeModel):
        def predict_proba(self, X):
            return np.full((len(X), 2), np.nan)
    posts = pd.DataFrame([{**dict.fromkeys(MODEL_FEATURES, 0), "post_id": 1}])
    with pytest.raises(ValueError, match="limit"):
        service.get_recommendations(posts, BadModel("bad"), 0)
    with pytest.raises(RuntimeError, match="вероятности"):
        service.get_recommendations(posts, BadModel("bad"), 5)


def test_startup_rejects_posts_without_text(monkeypatch, api_context):
    _, models, snapshot, posts = api_context
    tables = snapshot_export_tables(snapshot)
    tables["posts_info"] = posts.loc[posts.post_id != 13, ["post_id", "text", "topic"]]
    def fake_load_sql(sql, dtypes=None):
        for name in tables:
            if name == "posts_info":
                if "FROM public.post_text_df" in sql:
                    return tables[name].copy()
            elif f'FROM public."{name}"' in sql:
                return tables[name].copy()
        raise AssertionError("Неожиданный SQL")
    monkeypatch.setattr(service, "load_sql", fake_load_sql)
    with pytest.raises(RuntimeError, match="Нет текста"):
        service.initialize_service()
