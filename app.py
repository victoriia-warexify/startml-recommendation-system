"""API рекомендаций для двух моделей: HGB (control) и MLP (test)."""

import hashlib
import json
import os
import pickle
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from database import postgres_connection, feature_table
from features import (
    add_similarity_features,
    cast_feature_types,
    validate_model_snapshot,
    FEATURE_DEFINITION,
)

from fastapi import FastAPI, HTTPException, Query
from loguru import logger


from schema import PostGet, Response


# ============================================================
# PostgreSQL
# ============================================================

def load_sql(
    query: str,
    dtypes: Dict[str, Any] = None
) -> pd.DataFrame:
    """Читает SQL-запрос и закрывает соединение с PostgreSQL."""
    connection = postgres_connection()

    try:
        return pd.read_sql(
            query,
            connection,
            dtype=dtypes
        )
    finally:
        connection.close()


# ============================================================
# Модель
# ============================================================

EXP_SALT = "my_ab_test_salt"
CONTROL_PERCENT = 50

def load_model(model_version: str = "default") -> Any:
    """Загружает локальную модель из доверенного PKL-файла."""
    # Определяем директорию с моделями
    if os.environ.get("IS_LMS", "0") == "1":
        models_dir = os.environ.get("MODELS_DIR", ".")
    else:
        models_dir = os.environ.get("MODELS_DIR", str(Path(__file__).resolve().parent))

    # Формируем путь к файлу модели с префиксом model_
    model_filename = f"model_{model_version}.pkl"
    model_path = os.path.join(models_dir, model_filename)

    logger.info(f"Загрузка модели версии '{model_version}' из файла {model_path}...")

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл модели версии '{model_version}' не найден: {model_path}")
    except Exception:
        raise RuntimeError(
            f"Не удалось загрузить модель версии '{model_version}'. "
            "Проверьте доверие к PKL и версии зависимостей."
        ) from None

    logger.success(f"Модель версии '{model_version}' успешно загружена")

    return model

def get_exp_group(user_id: int) -> str:
    """Закрепляет пользователя за A/B-группой по хешу ID."""
    value = f"{EXP_SALT}_{user_id}"
    hash_value = hashlib.md5(
        value.encode()
    ).hexdigest()

    hash_int = int(
        hash_value[:8],
        16
    )

    bucket = hash_int % 100

    if bucket < CONTROL_PERCENT:
        return "control"

    return "test"

# ============================================================
# Просмотренные посты
# ============================================================

def normalize_timestamp(value) -> pd.Timestamp:
    """Приводит дату к UTC и убирает часовой пояс."""
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)

    return timestamp


def get_seen_post_ids(
    user_id: int,
    dt: datetime
) -> set[int]:
    """Возвращает посты, просмотренные до dt в пределах сохранённой истории."""
    if user_id not in user_seen_by_id.index:
        return set()

    serialized = user_seen_by_id.loc[
        user_id,
        "seen_history_json"
    ]

    if isinstance(serialized, pd.Series):
        serialized = serialized.iloc[0]

    if not serialized:
        return set()

    cutoff = normalize_timestamp(dt)

    try:
        history = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Некорректная история просмотров пользователя") from exc

    seen = set()

    if not isinstance(history, list):
        raise ValueError("Некорректная история просмотров пользователя")
    try:
        for post_id, timestamp_value in history:
            timestamp = normalize_timestamp(timestamp_value)
            if timestamp < cutoff:
                seen.add(int(post_id))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Некорректная история просмотров пользователя") from exc

    return seen


# ============================================================
# Инициализация
# ============================================================

# Переменные будут заполнены при старте API, а не при импорте модуля.
model_control = model_test = None
CONTROL_FEATURES = TEST_FEATURES = []
user_features = post_features = user_topic_features = user_cluster_features = pd.DataFrame()
user_seen_features = posts_info = pd.DataFrame()
user_features_by_id = user_seen_by_id = pd.DataFrame()
user_topic_features_by_id = {}
user_cluster_features_by_id = {}


def initialize_service():
    """Загружает модели и снимок признаков в память при запуске API."""
    global model_control, model_test, CONTROL_FEATURES, TEST_FEATURES
    global user_features, post_features, user_topic_features, user_cluster_features, user_seen_features
    global posts_info, user_topic_features_by_id, user_cluster_features_by_id, user_features_by_id, user_seen_by_id

    logger.info("Инициализация сервиса")

    model_control = load_model("control")
    model_test = load_model("test")

    CONTROL_FEATURES = list(model_control.feature_names_in_)
    TEST_FEATURES = list(model_test.feature_names_in_)

    user_features = load_sql(
        f"""
        SELECT *
        FROM public."{feature_table("user_features")}"
        """
    )

    post_features = load_sql(
        f"""
        SELECT *
        FROM public."{feature_table("post_features")}"
        """
    )

    user_topic_features = load_sql(
        f"""
        SELECT *
        FROM public."{feature_table("user_topic_features")}"
        """
    )

    user_cluster_features = load_sql(
        f"""
        SELECT *
        FROM public."{feature_table("user_cluster_features")}"
        """
    )

    user_seen_features = load_sql(
        f"""
        SELECT *
        FROM public."{feature_table("user_seen_features")}"
        """
    )

    posts_info = load_sql(
        """
        SELECT post_id, text, topic
        FROM public.post_text_df
        ORDER BY post_id
        """
    )



    tables = validate_model_snapshot(
        [model_control, model_test],
        {
            "user_features": user_features,
            "post_features": post_features,
            "user_topic_features": user_topic_features,
            "user_cluster_features": user_cluster_features,
            "user_seen_features": user_seen_features,
        }
    )
    user_features = tables["user_features"]
    post_features = tables["post_features"]
    user_topic_features = tables["user_topic_features"]
    user_cluster_features = tables["user_cluster_features"]
    user_seen_features = tables["user_seen_features"]


    user_features["user_id"] = (
        user_features["user_id"]
        .astype(int)
    )

    post_features["post_id"] = (
        post_features["post_id"]
        .astype(int)
    )

    post_features["topic"] = (
        post_features["topic"]
        .fillna("unknown")
        .astype(str)
    )

    user_topic_features["user_id"] = (
        user_topic_features["user_id"]
        .astype(int)
    )

    user_topic_features["topic"] = (
        user_topic_features["topic"]
        .fillna("unknown")
        .astype(str)
    )

    user_cluster_features["user_id"] = user_cluster_features["user_id"].astype(int)
    user_cluster_features["text_cluster"] = (
        user_cluster_features["text_cluster"].fillna("unknown").astype(str)
    )
    post_features["text_cluster"] = (
        post_features["text_cluster"].fillna("unknown").astype(str)
    )

    user_seen_features["user_id"] = (
        user_seen_features["user_id"]
        .astype(int)
    )

    posts_info["post_id"] = (
        posts_info["post_id"]
        .astype(int)
    )


    # Все кандидаты должны иметь текст и тему для ответа API.
    missing_posts = set(post_features["post_id"]) - set(posts_info["post_id"])
    if missing_posts:
        raise RuntimeError(f"Нет текста для {len(missing_posts)} постов из каталога признаков")

    for frame, keys in [
        (user_features, ["user_id"]),
        (post_features, ["post_id"]),
        (user_topic_features, ["user_id", "topic"]),
        (user_cluster_features, ["user_id", "text_cluster"]),
        (user_seen_features, ["user_id"]),
        (posts_info, ["post_id"]),
    ]:
        if frame.duplicated(keys).any():
            raise RuntimeError("Повторяющиеся ключи в таблицах признаков или постов")

    user_topic_features_by_id = {
        user_id: group
        for user_id, group in user_topic_features.groupby("user_id")
    }

    user_cluster_features_by_id = {
        user_id: group
        for user_id, group in user_cluster_features.groupby("user_id")
    }

    user_features_by_id = (
        user_features
        .set_index("user_id", drop=False)
    )

    user_seen_by_id = (
        user_seen_features
        .set_index("user_id", drop=False)
    )


    available = set(post_features.columns) | set(user_features.columns) | {
        "hour", "day_of_week", "month", "is_weekend",
        "user_topic_views", "user_topic_likes", "user_topic_ctr",
        "user_cluster_views", "user_cluster_likes", "user_cluster_ctr",
        "positive_cosine", "positive_dot", "positive_l2",
        "negative_cosine", "negative_dot", "negative_l2",
        "preference_margin", "distance_margin",
    }
    for columns in [CONTROL_FEATURES, TEST_FEATURES]:
        missing = set(columns) - available
        if missing:
            raise RuntimeError(
                f"Таблицы несовместимы с моделью, отсутствуют признаки: {sorted(missing)}"
            )

    logger.success("Сервис инициализирован")


@asynccontextmanager
async def lifespan(application):
    try:
        initialize_service()
    except Exception:
        # Не выводим ошибку драйвера: она может содержать реквизиты подключения.
        raise RuntimeError(
            "Не удалось запустить API. Проверьте PG*, MODELS_DIR, "
            "доступ к таблицам и совместимость моделей."
        ) from None
    yield


app = FastAPI(lifespan=lifespan)

def get_recommendations(
    candidates: pd.DataFrame,
    model: Any,
    limit: int
) -> pd.DataFrame:
    """Ранжирует уникальные посты по вероятности положительного отклика."""
    if limit < 1:
        raise ValueError("limit должен быть положительным")
    if candidates.empty:
        return pd.DataFrame(columns=["post_id", "score"])

    candidates = candidates.drop_duplicates("post_id").copy()

    X_candidates = candidates.drop(
        columns=[
            "user_id",
            "post_id",
            "timestamp",
            "score"
        ],
        errors="ignore"
    )

    expected_columns = list(model.feature_names_in_)

    missing_columns = [
        column
        for column in expected_columns
        if column not in X_candidates.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=(
                "Отсутствуют признаки модели: "
                f"{missing_columns}"
            )
        )

    X_candidates = X_candidates[
        expected_columns
    ]

    proba = np.asarray(model.predict_proba(X_candidates))
    if proba.shape != (len(X_candidates), 2) or not np.isfinite(proba).all():
        raise RuntimeError("Модель вернула некорректные вероятности")
    candidates["score"] = proba[:, 1]

    return (
        candidates[
            ["post_id", "score"]
        ]
        .sort_values(
            ["score", "post_id"],
            ascending=[False, True]
        )
        .head(limit)
    )


def get_recommendations_control(
    candidates: pd.DataFrame,
    limit: int
) -> pd.DataFrame:
    """Рекомендации контрольной модели HGB."""
    return get_recommendations(
        candidates=candidates,
        model=model_control,
        limit=limit
    )


def get_recommendations_test(
    candidates: pd.DataFrame,
    limit: int
) -> pd.DataFrame:
    """Рекомендации тестовой модели MLP."""
    return get_recommendations(
        candidates=candidates,
        model=model_test,
        limit=limit
    )


# ============================================================
# Endpoint
# ============================================================

@app.get(
    "/post/recommendations/",
    response_model=Response
)
def recommended_posts(
    user_id: int,
    dt: datetime,
    limit: int = Query(
        default=10,
        ge=1,
        le=100
    )
) -> Response:
    """Возвращает топ постов и группу эксперимента для пользователя."""
    if user_id not in user_features_by_id.index:
        raise HTTPException(
            status_code=404,
            detail=f"Пользователь {user_id} не найден"
        )

    if getattr(model_control, "feature_definition_", None) == FEATURE_DEFINITION:
        # Исторические признаки не обновляются после history_cutoff_.
        dt = normalize_timestamp(dt).to_pydatetime()
        if pd.Timestamp(dt) <= pd.Timestamp(model_control.history_cutoff_):
            raise HTTPException(
                status_code=422,
                detail="dt должен быть позже границы истории модели",
            )

    user_row = user_features_by_id.loc[
        user_id
    ]

    if isinstance(user_row, pd.DataFrame):
        user_row = user_row.iloc[0]

    candidates = post_features.copy()
    if candidates.empty:
        return Response(exp_group=get_exp_group(user_id), recommendations=[])

    # Исключаем посты, просмотренные до dt.
    seen_posts = get_seen_post_ids(
        user_id=user_id,
        dt=dt
    )

    if seen_posts:
        candidates = candidates[
            ~candidates["post_id"].isin(seen_posts)
        ].copy()

    candidates["hour"] = dt.hour
    candidates["day_of_week"] = dt.weekday()
    candidates["month"] = dt.month
    candidates["is_weekend"] = int(
        dt.weekday() in [5, 6]
    )

    for column in user_features.columns:
        candidates[column] = user_row[column]

    current_topic_features = (
        user_topic_features_by_id.get(
            user_id,
            pd.DataFrame(
                columns=[
                    "user_id",
                    "topic",
                    "user_topic_views",
                    "user_topic_likes",
                    "user_topic_ctr"
                ]
            )
        )[
            [
                "topic",
                "user_topic_views",
                "user_topic_likes",
                "user_topic_ctr"
            ]
        ]
    )

    candidates = candidates.merge(
        current_topic_features,
        on="topic",
        how="left",
        validate="many_to_one"
    )

    candidates["user_topic_views"] = (
        pd.to_numeric(candidates["user_topic_views"], errors="coerce")
        .fillna(0)
        .astype(np.int32)
    )

    candidates["user_topic_likes"] = (
        pd.to_numeric(candidates["user_topic_likes"], errors="coerce")
        .fillna(0)
        .astype(np.int32)
    )

    candidates["user_topic_ctr"] = (
        pd.to_numeric(candidates["user_topic_ctr"], errors="coerce")
        .fillna(candidates["topic_ctr"])
        .astype(np.float32)
    )

    current_cluster_features = user_cluster_features_by_id.get(
        user_id,
        pd.DataFrame(columns=[
            "user_id", "text_cluster", "user_cluster_views",
            "user_cluster_likes", "user_cluster_ctr"
        ])
    )[[
        "text_cluster", "user_cluster_views",
        "user_cluster_likes", "user_cluster_ctr"
    ]]
    candidates = candidates.merge(
        current_cluster_features,
        on="text_cluster",
        how="left",
        validate="many_to_one"
    )
    for column in ("user_cluster_views", "user_cluster_likes"):
        candidates[column] = (
            pd.to_numeric(candidates[column], errors="coerce")
            .fillna(0)
            .astype(np.int32)
        )
    candidates["user_cluster_ctr"] = (
        pd.to_numeric(candidates["user_cluster_ctr"], errors="coerce")
        .fillna(candidates["cluster_ctr"])
        .astype(np.float32)
    )

    candidates = add_similarity_features(candidates)

    candidates = cast_feature_types(candidates)

    exp_group = get_exp_group(user_id)

    if exp_group == "control":
        logger.info(
            f"user_id={user_id}: группа control, используется model_control"
        )

        top_posts = get_recommendations_control(
            candidates,
            limit
        )
    else:
        logger.info(
            f"user_id={user_id}: группа test, используется model_test"
        )

        top_posts = get_recommendations_test(
            candidates,
            limit
        )

    top_posts = top_posts.merge(
        posts_info,
        on="post_id",
        how="inner",
        validate="one_to_one"
    )

    result = []

    for _, row in top_posts.iterrows():
        topic = (
            "unknown"
            if pd.isna(row["topic"])
            else str(row["topic"])
        )

        result.append(
            PostGet(
                id=int(row["post_id"]),
                text=str(row["text"]),
                topic=topic
            )
        )

    return Response(
        exp_group=exp_group,
        recommendations=result
    )