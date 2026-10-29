"""Короткие тесты для временных признаков без подключения к PostgreSQL."""

import pandas as pd
import pytest

from features import (
    SPLIT_DATE,
    build_history_snapshot,
    build_temporal_datasets,
    temporal_split_masks,
)


def sample_data():
    users = pd.DataFrame([
        {
            "user_id": 1, "gender": 0, "age": 25, "country": "RU",
            "city": "Moscow", "exp_group": 1, "os": "iOS", "source": "ads",
        }
    ])
    posts = pd.DataFrame([
        {"post_id": 10, "topic": "tech", "text_cluster": "tech:c00", "text_length": 50, "word_count": 10},
        {"post_id": 11, "topic": "food", "text_cluster": "food:c00", "text_length": 80, "word_count": 15},
    ])
    feed = pd.DataFrame([
        {"user_id": 1, "post_id": 10, "timestamp": "2021-11-01 10:00", "action": "view", "target": 0},
        {"user_id": 1, "post_id": 10, "timestamp": "2021-11-01 10:05", "action": "like", "target": None},
        {"user_id": 1, "post_id": 11, "timestamp": "2021-11-01 11:00", "action": "view", "target": 1},
        {"user_id": 1, "post_id": 11, "timestamp": "2021-12-13 11:00", "action": "view", "target": 0},
    ])
    return users, posts, feed


def test_history_is_only_from_past():
    users, posts, feed = sample_data()
    train, test, _ = build_temporal_datasets(users, posts, feed)
    assert len(train) == 2
    assert len(test) == 1
    assert train.loc[0, "user_actions_count"] == 0
    assert train.loc[0, "post_views"] == 0
    assert train.loc[1, "user_views_count"] == 1
    assert train.loc[1, "user_likes_count"] == 1
    assert train.loc[1, "post_likes"] == 0
    assert test.loc[0, "user_actions_count"] == 3


def test_events_at_same_timestamp_do_not_see_each_other():
    users, posts, feed = sample_data()
    feed.loc[1, "timestamp"] = feed.loc[0, "timestamp"]
    train, _, _ = build_temporal_datasets(users, posts, feed)
    assert train.loc[0, "user_likes_count"] == 0
    assert train.loc[1, "user_likes_count"] == 1


def test_snapshot_accepts_history_without_future_views():
    users, posts, feed = sample_data()
    history_only = feed[pd.to_datetime(feed["timestamp"]) <= SPLIT_DATE].copy()
    snapshot = build_history_snapshot(users, posts, history_only)
    assert snapshot["user_features"].loc[0, "user_actions_count"] == 3
    assert snapshot["post_features"].loc[0, "post_likes"] == 1
    assert snapshot["post_features"].attrs["history_cutoff"] == SPLIT_DATE.isoformat()


def test_train_test_split_still_requires_both_sides():
    _, _, feed = sample_data()
    history_only = feed[pd.to_datetime(feed["timestamp"]) <= SPLIT_DATE].copy()
    with pytest.raises(ValueError, match="по обе стороны"):
        temporal_split_masks(history_only)


def test_snapshot_same_with_or_without_future_data():
    users, posts, feed = sample_data()
    history_only = feed[pd.to_datetime(feed["timestamp"]) <= SPLIT_DATE].copy()
    full_snapshot = build_history_snapshot(users, posts, feed)
    old_snapshot = build_history_snapshot(users, posts, history_only)
    for name in full_snapshot:
        pd.testing.assert_frame_equal(full_snapshot[name], old_snapshot[name])


def test_changing_view_targets_does_not_change_features():
    """Метка просмотра не должна попадать в исторические признаки."""
    users, posts, feed = sample_data()
    changed = feed.copy()
    view_mask = changed.action == "view"
    changed.loc[view_mask, "target"] = 1 - changed.loc[view_mask, "target"].astype(int)

    train1, test1, snapshot1 = build_temporal_datasets(users, posts, feed)
    train2, test2, snapshot2 = build_temporal_datasets(users, posts, changed)
    pd.testing.assert_frame_equal(train1.drop(columns="target"), train2.drop(columns="target"))
    pd.testing.assert_frame_equal(test1.drop(columns="target"), test2.drop(columns="target"))
    for name in snapshot1:
        pd.testing.assert_frame_equal(snapshot1[name], snapshot2[name])


def test_future_like_does_not_change_snapshot():
    """Лайки после границы Train не влияют на фиксированные признаки."""
    users, posts, feed = sample_data()
    future_like = pd.DataFrame([{
        "user_id": 1, "post_id": 11, "timestamp": "2021-12-14 18:00",
        "action": "like", "target": 0,
    }])
    with_future = feed.copy()
    with_future.loc[len(with_future)] = future_like.iloc[0]
    original = build_history_snapshot(users, posts, feed)
    altered = build_history_snapshot(users, posts, with_future)
    for name in original:
        pd.testing.assert_frame_equal(original[name], altered[name])


def test_model_snapshot_metadata_validation():
    """Проверяем совместимость модели и таблиц без обращения к БД."""
    from types import SimpleNamespace
    from features import mark_temporal_model, snapshot_export_tables, validate_model_snapshot

    users, posts, feed = sample_data()
    snapshot = build_history_snapshot(users, posts, feed)
    tables = snapshot_export_tables(snapshot)
    model = mark_temporal_model(SimpleNamespace(), snapshot)

    cleaned = validate_model_snapshot([model], tables)
    assert "_history_cutoff" not in cleaned["user_features"].columns

    model.history_cutoff_ = "2021-11-01T00:00:00"
    with pytest.raises(ValueError, match="разные даты"):
        validate_model_snapshot([model], tables)


def test_untagged_model_and_tables_are_not_accepted():
    """Нельзя обходить проверку среза с помощью старых артефактов."""
    from types import SimpleNamespace
    from features import validate_model_snapshot

    fake_tables = {
        "user_cluster_features": pd.DataFrame({"user_id": [1]}),
        "post_features": pd.DataFrame({"post_id": [10]}),
    }
    with pytest.raises(ValueError, match="старые модели"):
        validate_model_snapshot([SimpleNamespace()], fake_tables)
