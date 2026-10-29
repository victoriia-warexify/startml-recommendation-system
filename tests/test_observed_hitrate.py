"""Чистые проверки observed HitRate без доступа к PostgreSQL и обучению."""

import numpy as np
import pandas as pd
import pytest

from evaluate_observed_hitrate import observed_hitrate, probabilities_in_batches


def test_observed_hitrate_ranks_only_observed_events():
    rows = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2],
        "post_id": [11, 12, 13, 21, 22, 23],
        "target": [0, 1, 1, 1, 0, 0],
    })
    scores = {"model": np.array([0.9, 0.8, 0.2, 0.1, 0.9, 0.8])}
    report, detail = observed_hitrate(rows, scores, k=2)
    assert report["users"] == 2
    assert report["events"] == 6
    assert report["hitrate"]["model"] == 0.5
    assert report["oracle"] == 1.0
    # user1: шанс попадания при выборе 2 из 3 = 1;
    # user2: шанс выбрать единственную положительную строку = 2/3.
    assert report["random_expected_hitrate"] == pytest.approx(5 / 6)
    assert detail["hit_model"].tolist() == [1, 0]


def test_topk_any_positive_is_one_not_number_of_likes():
    rows = pd.DataFrame({
        "user_id": [1, 1, 1], "post_id": [3, 2, 1], "target": [1, 1, 0]
    })
    summary, _ = observed_hitrate(rows, {"A": [0.7, 0.6, 0.5]}, k=2)
    assert summary["hit_counts"]["A"] == 1
    assert summary["hitrate"]["A"] == 1.0


def test_ties_are_deterministic_and_repeated_views_counted():
    rows = pd.DataFrame({
        "user_id": [1, 1, 1], "post_id": [6, 6, 7], "target": [0, 1, 1]
    })
    summary, detail = observed_hitrate(rows, {"A": [0.5, 0.5, 0.5]}, k=2)
    assert summary["duplicate_user_post_views"] == 1
    assert summary["hitrate"]["A"] == 1.0
    assert detail["n_unique_posts"].tolist() == [2]


def test_bad_scores_are_rejected():
    rows = pd.DataFrame({"user_id": [1, 1], "post_id": [2, 3], "target": [0, 1]})
    with pytest.raises(ValueError, match="некорректные"):
        observed_hitrate(rows, {"A": [0.3, np.nan]})


class FakeModel:
    def predict_proba(self, X):
        p = X["p"].to_numpy(dtype=float)
        return np.column_stack([1-p, p])


def test_batches_preserve_row_order():
    data = pd.DataFrame({"p": [0.1, 0.2, 0.3, 0.4, 0.5]})
    result = probabilities_in_batches(FakeModel(), data, ["p"], batch_size=2)
    np.testing.assert_allclose(result, data["p"].to_numpy())

from evaluate_observed_hitrate import audit_unique_posts, verify_frozen_test_features


def test_duplicate_post_cannot_fill_two_top_slots():
    rows = pd.DataFrame({
        "user_id": [1, 1, 1],
        "post_id": [10, 10, 11],
        "timestamp": pd.to_datetime(["2021-12-13", "2021-12-14", "2021-12-15"]),
        "target": [0, 0, 1],
    })
    scores = {"HGB": [0.99, 0.98, 0.97]}
    original, _ = observed_hitrate(rows, scores, k=2)
    audit, by_user = audit_unique_posts(rows, scores, k=2)
    assert original["hitrate"]["HGB"] == 0.0
    assert audit["users_with_duplicates_in_topk"]["HGB"] == 1
    assert audit["extra_duplicate_topk_slots"]["HGB"] == 1
    assert audit["first_hitrate"]["HGB"] == 1.0
    assert audit["ever_hitrate"]["HGB"] == 1.0
    assert by_user.loc[0, "hit_unique_first_HGB"] == 1


def test_ever_positive_differs_from_first_view_target():
    rows = pd.DataFrame({
        "user_id": [2, 2, 2],
        "post_id": [10, 10, 11],
        "timestamp": pd.to_datetime(["2021-12-13", "2021-12-14", "2021-12-15"]),
        "target": [0, 1, 0],
    })
    audit, _ = audit_unique_posts(rows, {"MLP": [0.9, 0.2, 0.1]}, k=1)
    assert audit["first_hitrate"]["MLP"] == 0.0
    assert audit["ever_hitrate"]["MLP"] == 1.0


def test_first_score_uses_earliest_timestamp_not_first_dataframe_row():
    rows = pd.DataFrame({
        "user_id": [1, 1, 1],
        "post_id": [4, 4, 5],
        "timestamp": pd.to_datetime(["2021-12-15", "2021-12-13", "2021-12-14"]),
        "target": [0, 0, 1],
    })
    # Первый просмотр 4 имеет score 0.1, а более поздний — 0.99.
    audit, _ = audit_unique_posts(rows, {"HGB": [0.99, 0.1, 0.8]}, k=1)
    assert audit["first_hitrate"]["HGB"] == 1.0


class HourModel:
    def predict_proba(self, X):
        p = X["hour"].to_numpy(dtype=float) / 24
        return np.column_stack([1 - p, p])


def test_fixed_time_scoring_changes_only_calendar_values():
    events = pd.DataFrame({"hour": [2, 7, 12], "age": [30, 30, 30]})
    original = probabilities_in_batches(HourModel(), events, ["hour", "age"], 2)
    fixed = probabilities_in_batches(
        HourModel(), events, ["hour", "age"], 2,
        fixed_time=pd.Timestamp("2021-12-12 05:00:01"),
    )
    np.testing.assert_allclose(original, np.array([2, 7, 12]) / 24)
    np.testing.assert_allclose(fixed, np.array([5, 5, 5]) / 24)
    assert events["hour"].tolist() == [2, 7, 12]


def test_snapshot_features_ignore_future_timestamp_and_test_target():
    from features import (
        SPLIT_DATE, USER_COLUMNS, USER_HISTORY_COLUMNS,
        POST_STATIC_COLUMNS, POST_HISTORY_COLUMNS, POST_EXTRA_COLUMNS,
        USER_TOPIC_COLUMNS, USER_CLUSTER_COLUMNS,
    )
    user_vals = dict.fromkeys(USER_COLUMNS + USER_HISTORY_COLUMNS, 1)
    user_vals.update(country="RU", city="Kazan", os="iOS", source="ads")
    users = pd.DataFrame([{"user_id": 1, **user_vals}])
    post_vals = dict.fromkeys(POST_STATIC_COLUMNS + POST_HISTORY_COLUMNS + POST_EXTRA_COLUMNS, 1)
    post_vals.update(text_cluster="news:c00", post_ctr=0.2, topic_ctr=0.2, cluster_ctr=0.2)
    posts = pd.DataFrame([{"post_id": 7, "topic": "news", **post_vals}])
    posts.attrs["history_cutoff"] = SPLIT_DATE.isoformat()
    snapshot = {
        "user_features": users,
        "post_features": posts,
        "user_topic_features": pd.DataFrame(columns=["user_id", "topic", *USER_TOPIC_COLUMNS]),
        "user_cluster_features": pd.DataFrame(columns=["user_id", "text_cluster", *USER_CLUSTER_COLUMNS]),
    }
    views = pd.DataFrame({
        "user_id": [1, 1], "post_id": [7, 7],
        "timestamp": pd.to_datetime(["2021-12-13 12:00:00", "2021-12-16 08:00:00"]),
        "action": ["view", "view"], "target": [0, 1],
    })
    report = verify_frozen_test_features(views, snapshot)
    assert report == {"checked_rows": 2, "fixed_features": 28, "time_features": 4}


def test_unique_hitrate_averages_across_users_not_events():
    """Пользователи имеют одинаковый вес независимо от числа просмотров."""
    from evaluate_observed_hitrate import audit_unique_posts

    rows = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 2, 2],
        "post_id": [10, 11, 12, 13, 20, 21],
        "timestamp": pd.to_datetime(["2021-12-13"] * 6),
        "target": [0, 1, 1, 1, 1, 0],
    })
    summary, by_user = audit_unique_posts(
        rows, {"M": [0.1, 0.8, 0.7, 0.6, 0.1, 0.9]}, k=1
    )
    assert summary["users"] == 2
    assert summary["first_hitrate"]["M"] == 0.5
    assert by_user["hit_unique_first_M"].tolist() == [1, 0]


def test_unique_hitrate_handles_k_larger_than_unique_posts():
    from evaluate_observed_hitrate import audit_unique_posts

    rows = pd.DataFrame({
        "user_id": [1, 1, 2],
        "post_id": [10, 10, 20],
        "timestamp": pd.to_datetime(["2021-12-13", "2021-12-14", "2021-12-15"]),
        "target": [0, 1, 1],
    })
    summary, detail = audit_unique_posts(rows, {"M": [0.8, 0.9, 0.2]}, k=5)
    assert summary["first_hitrate"]["M"] == 0.5
    assert summary["ever_hitrate"]["M"] == 1.0
    assert detail["n_unique_posts"].tolist() == [1, 1]


def test_metric_csv_primary_is_unique_and_event_csv_is_separate(tmp_path):
    from evaluate_observed_hitrate import save_metric_details

    unique = pd.DataFrame({"user_id": [1], "hit_unique_first_HGB": [1]})
    events = pd.DataFrame({"user_id": [1], "hit_HGB": [0]})
    output, event_path = save_metric_details(unique, events, tmp_path / "results.csv")

    assert output.name == "results.csv"
    assert event_path.name == "results_events.csv"
    pd.testing.assert_frame_equal(pd.read_csv(output), unique)
    pd.testing.assert_frame_equal(pd.read_csv(event_path), events)


def test_unique_hitrate_rejects_nan_scores():
    from evaluate_observed_hitrate import audit_unique_posts

    rows = pd.DataFrame({
        "user_id": [1, 1], "post_id": [10, 11],
        "timestamp": pd.to_datetime(["2021-12-13", "2021-12-14"]),
        "target": [0, 1],
    })
    with pytest.raises(ValueError, match="некорректные"):
        audit_unique_posts(rows, {"MLP": [0.1, np.nan]}, k=1)
