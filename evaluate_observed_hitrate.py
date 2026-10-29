
"""Диагностический Observed HitRate@K на просмотренных постах Test.

Основная метрика:
- один post_id учитывается один раз для каждого пользователя;
- score и target берутся с первого просмотра;
- для каждого пользователя выбираются K постов с наибольшим score;
- HitRate = доля пользователей, у которых в топ-K есть target=1.

Кандидаты взяты из реально наблюдавшихся будущих просмотров.
Поэтому метрика НЕ равна онлайн HitRate или оценке LMS.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42

COURSE_CATBOOST_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "gender", "age", "country", "city", "exp_group",
    "os", "source", "topic", "text_cluster",
    "text_length", "word_count",
]


def model_feature_names(model):
    """Возвращает входные признаки модели."""
    names = getattr(model, "feature_names_in_", None)

    if names is None:
        names = getattr(model, "feature_names_", None)

    if names is None or len(names) == 0:
        raise ValueError(
            "У модели отсутствует список входных признаков"
        )

    return list(names)


def check_model_snapshot(models: dict, snapshot: dict):
    """Проверяет совместимость моделей и исторических признаков."""
    metadata = snapshot["post_features"].attrs

    baseline = model_feature_names(models["HGB"])
    available = set(baseline)

    for name, model in models.items():
        feature_names = model_feature_names(model)

        missing = [
            feature
            for feature in feature_names
            if feature not in available
        ]

        if missing:
            raise ValueError(
                f"{name}: отсутствуют признаки: {missing}"
            )

        if name == "CatBoost":
            if feature_names != COURSE_CATBOOST_FEATURES:
                raise ValueError(
                    "CatBoost: неожиданный набор признаков"
                )

            expected_categorical = [
                "country", "city", "os",
                "source", "topic", "text_cluster",
            ]

            actual_categorical = [
                feature_names[i]
                for i in model.get_cat_feature_indices()
            ]

            if actual_categorical != expected_categorical:
                raise ValueError(
                    "CatBoost: несовместимые категориальные признаки"
                )

            print(
                "Внимание: CatBoost может использовать "
                "другое определение текстовых признаков."
            )
            continue

        if feature_names != baseline:
            raise ValueError(
                f"{name}: отличается список или порядок признаков"
            )

        if (
            getattr(model, "history_cutoff_", None)
            != metadata["history_cutoff"]
        ):
            raise ValueError(
                f"{name}: неверная дата исторического snapshot"
            )

        if (
            getattr(model, "feature_definition_", None)
            != metadata["feature_definition"]
        ):
            raise ValueError(
                f"{name}: несовместимая версия признаков"
            )


def observed_hitrate(
    events: pd.DataFrame,
    scores: dict[str, np.ndarray],
    k: int = 5,
):
    """Дополнительная диагностика по всем строкам просмотров.

    В отличие от основной метрики, здесь один post_id
    может встречаться несколько раз.
    """
    if k < 1:
        raise ValueError("k должен быть положительным")

    if events.empty:
        raise ValueError("Нет просмотров Test")

    needed = {"user_id", "post_id", "target"}

    if not needed.issubset(events.columns):
        raise ValueError(
            f"Отсутствуют столбцы: {needed - set(events.columns)}"
        )

    if (
        events["target"].isna().any()
        or not events["target"].isin([0, 1]).all()
    ):
        raise ValueError("target должен быть 0 или 1")

    base = (
        events[["user_id", "post_id", "target"]]
        .reset_index(drop=True)
        .copy()
    )

    base["_row"] = np.arange(len(base), dtype=np.int64)

    detail = base.groupby("user_id", sort=True).agg(
        n_test_views=("target", "size"),
        n_positive_views=("target", "sum"),
        n_unique_posts=("post_id", "nunique"),
        oracle_hit=("target", "max"),
    )

    for name, values in scores.items():
        values = np.asarray(values, dtype=np.float64)

        if (
            values.ndim != 1
            or len(values) != len(base)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"{name}: некорректные предсказания"
            )

        ranked = base.assign(_score=values).sort_values(
            ["user_id", "_score", "post_id", "_row"],
            ascending=[True, False, True, True],
            kind="stable",
        )

        top = ranked.groupby(
            "user_id", sort=False
        ).head(k)

        hits = top.groupby("user_id")["target"].max()

        detail[f"hit_{name}"] = (
            hits.reindex(detail.index, fill_value=0)
            .astype(np.int8)
        )

    n = detail["n_test_views"].to_numpy(
        dtype=np.float64
    )

    positives = detail["n_positive_views"].to_numpy(
        dtype=np.float64
    )

    negatives = n - positives

    miss_probability = np.ones(
        len(detail), dtype=np.float64
    )

    for position in range(k):
        active = n > position

        miss_probability[active] *= np.clip(
            (negatives[active] - position)
            / (n[active] - position),
            0.0,
            1.0,
        )

    random_expected = float(
        (1.0 - miss_probability).mean()
    )

    detail = detail.reset_index()

    summary = {
        "random_expected_hitrate": random_expected,
        "users": len(detail),
        "events": len(base),
        "duplicate_user_post_views": int(
            len(base)
            - len(base.drop_duplicates(
                ["user_id", "post_id"]
            ))
        ),
        "oracle": float(detail["oracle_hit"].mean()),
        "hitrate": {
            name: float(detail[f"hit_{name}"].mean())
            for name in scores
        },
        "hit_counts": {
            name: int(detail[f"hit_{name}"].sum())
            for name in scores
        },
    }

    return summary, detail


def audit_unique_posts(
    events: pd.DataFrame,
    scores: dict[str, np.ndarray],
    k: int = 5,
):
    """Основная диагностика HitRate@K по уникальным постам.

    Для каждой пары user_id/post_id:
    - score берётся с первого просмотра;
    - first target — метка первого просмотра;
    - ever target — была ли положительная метка позднее.

    Основная метрика проекта — first_hitrate.
    """
    needed = {
        "user_id", "post_id", "timestamp", "target"
    }

    if not needed.issubset(events.columns):
        raise ValueError(
            f"Отсутствуют столбцы: {needed - set(events.columns)}"
        )

    if k < 1 or events.empty:
        raise ValueError(
            "Для оценки нужны события и k >= 1"
        )

    if events["timestamp"].isna().any():
        raise ValueError(
            "timestamp содержит пропуски"
        )

    if (
        events["target"].isna().any()
        or not events["target"].isin([0, 1]).all()
    ):
        raise ValueError("target должен быть 0 или 1")

    base = (
        events[
            ["user_id", "post_id", "timestamp", "target"]
        ]
        .reset_index(drop=True)
        .copy()
    )

    base["_row"] = np.arange(
        len(base), dtype=np.int64
    )

    pair = ["user_id", "post_id"]

    chronology = base.sort_values(
        [*pair, "timestamp", "_row"],
        kind="stable",
    )

    first = chronology.drop_duplicates(
        pair, keep="first"
    ).copy()

    ever_target = (
        base.groupby(pair, sort=False)["target"]
        .max()
    )

    first["ever_target"] = (
        first.set_index(pair)
        .index.map(ever_target)
        .astype(np.int8)
    )

    first_indices = first["_row"].to_numpy(
        dtype=np.int64
    )

    detail = base.groupby(
        "user_id", sort=True
    ).agg(
        n_views=("target", "size"),
        n_unique_posts=("post_id", "nunique"),
    )

    detail["n_repeated_views"] = (
        detail["n_views"] - detail["n_unique_posts"]
    )

    summary = {
        "users": len(detail),
        "repeated_rows": int(
            len(base) - len(first)
        ),
        "users_with_repeated_posts": int(
            (detail["n_repeated_views"] > 0).sum()
        ),
        "first_hitrate": {},
        "ever_hitrate": {},
        "users_with_duplicates_in_topk": {},
        "extra_duplicate_topk_slots": {},
    }

    for name, values in scores.items():
        values = np.asarray(
            values, dtype=np.float64
        )

        if (
            values.ndim != 1
            or len(values) != len(base)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"{name}: некорректные предсказания"
            )

        # Проверяем повторы среди всех просмотров.
        raw_ranked = base.assign(
            _score=values
        ).sort_values(
            ["user_id", "_score", "post_id", "_row"],
            ascending=[True, False, True, True],
            kind="stable",
        )

        raw_top = raw_ranked.groupby(
            "user_id", sort=False
        ).head(k)

        dup_slots = (
            raw_top.groupby("user_id")["post_id"].size()
            - raw_top.groupby("user_id")["post_id"].nunique()
        )

        detail[f"duplicate_slots_{name}"] = (
            dup_slots.reindex(
                detail.index, fill_value=0
            )
            .astype(np.int32)
        )

        # Один post_id — одна позиция в ранжировании.
        # Используем score первого просмотра.
        unique_ranked = first.assign(
            _score=values[first_indices]
        ).sort_values(
            ["user_id", "_score", "post_id", "_row"],
            ascending=[True, False, True, True],
            kind="stable",
        )

        top = unique_ranked.groupby(
            "user_id", sort=False
        ).head(k)

        for target_name, column in (
            ("first", "target"),
            ("ever", "ever_target"),
        ):
            hits = top.groupby(
                "user_id"
            )[column].max()

            key = (
                f"hit_unique_{target_name}_{name}"
            )

            detail[key] = (
                hits.reindex(
                    detail.index, fill_value=0
                )
                .astype(np.int8)
            )

            summary[
                f"{target_name}_hitrate"
            ][name] = float(detail[key].mean())

        summary[
            "users_with_duplicates_in_topk"
        ][name] = int(
            (
                detail[f"duplicate_slots_{name}"] > 0
            ).sum()
        )

        summary[
            "extra_duplicate_topk_slots"
        ][name] = int(
            detail[f"duplicate_slots_{name}"].sum()
        )

    return summary, detail.reset_index()


def verify_frozen_test_features(
    views: pd.DataFrame,
    snapshot: dict,
    max_rows: int = 64,
):
    """Проверяет отсутствие изменения исторических признаков Test."""
    from features import (
        MODEL_FEATURES,
        SPLIT_DATE,
        build_dataset_from_features,
    )

    if views.empty:
        raise ValueError("Нет просмотров")

    metadata = snapshot["post_features"].attrs

    if (
        pd.Timestamp(metadata.get("history_cutoff"))
        != pd.Timestamp(SPLIT_DATE)
    ):
        raise AssertionError(
            "Неверная дата исторического snapshot"
        )

    sample = views.iloc[:max_rows].copy()
    alternate = sample.copy()

    alternate["timestamp"] = (
        alternate["timestamp"]
        + pd.Timedelta(days=1, hours=1)
    )

    alternate["target"] = (
        1 - alternate["target"].astype(int)
    )

    tables = [
        snapshot["user_features"],
        snapshot["post_features"],
        snapshot["user_topic_features"],
        snapshot["user_cluster_features"],
    ]

    original_X = build_dataset_from_features(
        sample, *tables
    )

    altered_X = build_dataset_from_features(
        alternate, *tables
    )

    calendar = {
        "hour", "day_of_week", "month", "is_weekend"
    }

    fixed_columns = [
        feature
        for feature in MODEL_FEATURES
        if feature not in calendar
    ]

    pd.testing.assert_frame_equal(
        original_X[fixed_columns],
        altered_X[fixed_columns],
        check_exact=True,
    )

    assert (
        original_X["hour"] != altered_X["hour"]
    ).all()

    return {
        "checked_rows": len(sample),
        "fixed_features": len(fixed_columns),
        "time_features": len(calendar),
    }


def probabilities_in_batches(
    model,
    rows: pd.DataFrame,
    feature_names: list[str],
    batch_size: int,
    fixed_time: pd.Timestamp | None = None,
):
    """Предсказывает вероятности частями."""
    if batch_size < 1:
        raise ValueError(
            "batch_size должен быть положительным"
        )

    scores = np.empty(
        len(rows), dtype=np.float64
    )

    for start in range(0, len(rows), batch_size):
        stop = min(
            start + batch_size, len(rows)
        )

        X = rows.iloc[
            start:stop
        ][feature_names]

        if fixed_time is not None:
            X = X.copy()
            dt = pd.Timestamp(fixed_time)

            calendar_values = (
                ("hour", dt.hour),
                ("day_of_week", dt.dayofweek),
                ("month", dt.month),
                ("is_weekend", int(dt.dayofweek >= 5)),
            )

            for column, value in calendar_values:
                if column in X.columns:
                    X[column] = value

        proba = model.predict_proba(X)

        if proba.shape != (len(X), 2):
            raise ValueError(
                "Модель должна возвращать две вероятности"
            )

        scores[start:stop] = proba[:, 1]

    return scores


def save_metric_details(
    unique_detail: pd.DataFrame,
    event_detail: pd.DataFrame,
    output: str | Path,
) -> tuple[Path, Path]:
    """Сохраняет основной HitRate по постам и дополнительный по событиям."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    event_path = output.with_name(f"{output.stem}_events{output.suffix}")
    unique_detail.to_csv(output, index=False)
    event_detail.to_csv(event_path, index=False)
    return output, event_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--control",
        default="models_5pct/model_control.pkl",
    )

    parser.add_argument(
        "--test",
        default="models_5pct/model_test.pkl",
    )

    parser.add_argument(
        "--course-catboost",
        default=None,
    )

    parser.add_argument(
        "--user-mod",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--output",
        default="observed_hitrate_5pct.csv",
    )

    parser.add_argument(
        "--audit-fixed-time",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.k < 1
        or args.user_mod < 1
        or args.batch_size < 1
        or (
            args.max_users is not None
            and args.max_users < 1
        )
    ):
        parser.error(
            "Числовые параметры должны быть положительными"
        )

    from connection import load_data

    from features import (
        SPLIT_DATE,
        prepare_temporal_feed,
        build_post_features,
        build_history_snapshot,
        build_dataset_from_features,
    )

    models = {}

    model_paths = (
        ("HGB", args.control),
        ("MLP", args.test),
        ("CatBoost", args.course_catboost),
    )

    for label, path in model_paths:
        if path is None:
            continue

        path = Path(path)

        if not path.is_file():
            parser.error(
                f"Не найдена модель: {path}"
            )

        with path.open("rb") as file:
            models[label] = pickle.load(file)

    print(
        "Загружаем историю и готовим Test...",
        flush=True,
    )

    users, posts, feed = load_data(
        user_mod=args.user_mod
    )

    feed = prepare_temporal_feed(feed)

    history = feed.loc[
        feed["timestamp"] <= SPLIT_DATE
    ]

    views = feed.loc[
        (feed["timestamp"] > SPLIT_DATE)
        & (feed["action"] == "view")
    ]

    if views.empty:
        raise ValueError(
            "Нет просмотров в Test"
        )

    if args.max_users is not None:
        user_ids = views[
            "user_id"
        ].drop_duplicates()

        if len(user_ids) > args.max_users:
            selected = user_ids.sample(
                n=args.max_users,
                random_state=RANDOM_STATE,
            )

            views = views.loc[
                views["user_id"].isin(selected)
            ]

    static_posts = build_post_features(
        posts,
        fit_pca=False,
    )

    snapshot = build_history_snapshot(
        users,
        static_posts,
        history,
    )

    check_model_snapshot(models, snapshot)

    observed_test = build_dataset_from_features(
        views,
        snapshot["user_features"],
        snapshot["post_features"],
        snapshot["user_topic_features"],
        snapshot["user_cluster_features"],
    )

    print(
        f"Строк Test: {len(observed_test)}; "
        f"пользователей: {observed_test.user_id.nunique()}",
        flush=True,
    )

    stability = verify_frozen_test_features(
        views,
        snapshot,
    )

    print(
        "Проверка исторических признаков: OK "
        f"({stability['checked_rows']} строк)",
        flush=True,
    )

    scores = {}

    for name, model in models.items():
        print(
            f"Считаем вероятности: {name}",
            flush=True,
        )

        scores[name] = probabilities_in_batches(
            model,
            observed_test,
            model_feature_names(model),
            args.batch_size,
        )

    # Популярность на основе истории до Test.
    scores["popularity"] = observed_test[
        "post_likes"
    ].to_numpy(dtype=np.float64)

    summary, detail = observed_hitrate(
        observed_test,
        scores,
        k=args.k,
    )

    unique_summary, unique_detail = audit_unique_posts(
        observed_test,
        scores,
        k=args.k,
    )

    print(
        f"\nОсновная метрика: Observed HitRate@{args.k}"
    )
    print(
        "Уникальные посты, target первого просмотра"
    )

    for name, value in unique_summary[
        "first_hitrate"
    ].items():
        print(
            f"{name:12s}: {value:.6f}"
        )

    print(
        "\nДополнительная диагностика повторов:"
    )

    print(
        f"Повторных строк: "
        f"{unique_summary['repeated_rows']}"
    )

    for name in scores:
        print(
            f"{name}: повторных позиций в Top-K = "
            f"{unique_summary['extra_duplicate_topk_slots'][name]}"
        )

    if args.audit_fixed_time:
        asof = (
            SPLIT_DATE
            + pd.Timedelta(seconds=1)
        )

        fixed_scores = {}

        for name, model in models.items():
            fixed_scores[name] = probabilities_in_batches(
                model,
                observed_test,
                model_feature_names(model),
                args.batch_size,
                fixed_time=asof,
            )

        fixed_scores["popularity"] = (
            scores["popularity"].copy()
        )

        fixed_summary, fixed_detail = audit_unique_posts(
            observed_test,
            fixed_scores,
            k=args.k,
        )

        print(
            f"\nHitRate при фиксированном времени {asof}:"
        )

        for name in fixed_scores:
            print(
                f"{name:12s}: "
                f"{fixed_summary['first_hitrate'][name]:.6f}"
            )

            unique_detail[
                f"hit_fixed_time_{name}"
            ] = fixed_detail[
                f"hit_unique_first_{name}"
            ]

    print(
        "\nВнимание: это диагностическая offline-метрика, "
        "не онлайн HitRate и не LMS."
    )

    if args.output:
        # Основной файл соответствует метрике, напечатанной выше.
        output, event_path = save_metric_details(
            unique_detail, detail, args.output
        )
        print(f"\nObserved HitRate по уникальным постам: {output}")
        print(f"Дополнительная оценка по событиям: {event_path}")


if __name__ == "__main__":
    main()
