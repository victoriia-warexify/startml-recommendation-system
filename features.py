"""Подготовка текстовых и исторических признаков для HGB, MLP и API."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import groupby
from typing import List

import numpy as np
import pandas as pd


# ============================================================
# Настройки
# ============================================================

SPLIT_DATE = pd.Timestamp("2021-12-12")
RANDOM_STATE = 42

# Одинаковый набор и порядок полей для HGB, MLP и API.
MODEL_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "gender", "age", "country", "city", "exp_group", "os", "source", "topic",
    "user_actions_count", "user_post_count", "user_likes_count", "user_views_count",
    "post_views", "post_likes", "post_ctr", "post_popularity",
    "topic_views", "topic_likes", "topic_ctr",
    "user_topic_views", "user_topic_likes", "user_topic_ctr",
    "text_cluster", "text_length", "word_count",
    "user_cluster_views", "user_cluster_likes", "user_cluster_ctr",
]

DROP_COLUMNS = ["target", "user_id", "post_id", "timestamp"]
FEATURE_DEFINITION = "past_only_32_text_cluster_features_v1"

USER_COLUMNS = ["gender", "age", "country", "city", "exp_group", "os", "source"]
USER_HISTORY_COLUMNS = [
    "user_actions_count", "user_post_count", "user_likes_count", "user_views_count"
]
POST_HISTORY_COLUMNS = [
    "post_views", "post_likes", "post_ctr", "post_popularity",
    "topic_views", "topic_likes", "topic_ctr",
]
USER_TOPIC_COLUMNS = ["user_topic_views", "user_topic_likes", "user_topic_ctr"]
USER_CLUSTER_COLUMNS = ["user_cluster_views", "user_cluster_likes", "user_cluster_ctr"]
POST_STATIC_COLUMNS = ["text_cluster", "text_length", "word_count"]
# cluster_ctr — вспомогательная статистика для сглаживания user_cluster_ctr,
# в MODEL_FEATURES не входит.
POST_EXTRA_COLUMNS = ["cluster_ctr"]
CATEGORICAL_COLUMNS = {"country", "city", "os", "source", "topic", "text_cluster"}
FLOAT_COLUMNS = {
    "post_ctr", "post_popularity", "topic_ctr", "user_topic_ctr",
    "user_cluster_ctr", "cluster_ctr",
}
# Метаданные описывают формат и дату, но не идентифицируют выборку.
TABLE_METADATA = ["_feature_definition", "_history_cutoff"]


# ============================================================
# Общие функции
# ============================================================

def check_required_columns(
    dataframe: pd.DataFrame,
    required_columns: List[str],
    dataframe_name: str,
) -> None:
    """Проверяет наличие необходимых столбцов в таблице."""
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"В {dataframe_name} отсутствуют колонки: {missing}")


def normalize_event_times(values: pd.Series, errors: str = "raise") -> pd.Series:
    """Все timestamps сравниваются в UTC, затем представляются без tz."""
    return pd.to_datetime(values, utc=True, errors=errors).dt.tz_localize(None)


def validated_target(values: pd.Series) -> pd.Series:
    """Для view метка должна быть задана; для like target не используется."""
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError("target у action='view' должен быть равен 0 или 1 без пропусков")
    return values.astype(np.int8)


def cast_feature_types(frame: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Приводит общие признаки к одинаковым типам в обучении и API."""
    result = frame.copy() if copy else frame
    for column in result.columns:
        if column in CATEGORICAL_COLUMNS:
            result[column] = result[column].fillna("unknown").astype(str)
        elif column in FLOAT_COLUMNS:
            result[column] = result[column].astype(np.float32)
        elif column in USER_HISTORY_COLUMNS + USER_TOPIC_COLUMNS[:2] + USER_CLUSTER_COLUMNS[:2] + [
            "post_views", "post_likes", "topic_views", "topic_likes",
            "text_length", "word_count", "hour", "day_of_week", "month",
        ]:
            result[column] = result[column].astype(np.int32)
        elif column in {"target", "is_weekend"}:
            result[column] = result[column].astype(np.int8)
    return result


# ============================================================
# Временные признаки
# ============================================================

def build_time_features(feed_df: pd.DataFrame) -> pd.DataFrame:
    """Выделяет просмотры и добавляет признаки даты и времени."""
    check_required_columns(
        feed_df, ["user_id", "post_id", "timestamp", "action", "target"], "feed_df"
    )
    views = feed_df.loc[feed_df["action"] == "view"].copy()
    views["timestamp"] = normalize_event_times(views["timestamp"])
    views["target"] = validated_target(views["target"])
    views["hour"] = views["timestamp"].dt.hour
    views["day_of_week"] = views["timestamp"].dt.dayofweek
    views["month"] = views["timestamp"].dt.month
    views["is_weekend"] = views["day_of_week"].isin([5, 6]).astype(np.int8)
    return views[
        ["user_id", "post_id", "timestamp", "target", "hour", "day_of_week", "month", "is_weekend"]
    ]


# ============================================================
# Признаки постов
# ============================================================

def build_post_features(posts_df: pd.DataFrame, fit_pca: bool = False) -> pd.DataFrame:
    """Строит TF-IDF, SVD и текстовые кластеры внутри каждой темы.

    Использует только тексты постов и темы, без истории взаимодействий.
    """
    if fit_pca:
        raise ValueError("Текстовый PCA не используется: используется TruncatedSVD")

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    posts = posts_df.rename(columns={"id": "post_id"}) if "post_id" not in posts_df else posts_df
    check_required_columns(posts, ["post_id", "topic", "text"], "posts_df")
    if posts.empty:
        raise ValueError("В posts_df отсутствуют посты для кластеризации")
    if posts["post_id"].isna().any() or posts["post_id"].duplicated().any():
        raise ValueError("У постов должны быть уникальные post_id")

    result = posts[["post_id", "topic", "text"]].copy()
    result = result.sort_values("post_id", kind="stable").reset_index(drop=True)
    result["topic"] = result["topic"].fillna("unknown").astype(str)
    texts = result["text"].fillna("").astype(str)
    result["text_length"] = texts.str.len().astype(np.int32)
    result["word_count"] = texts.str.split().str.len().astype(np.int32)

    # Статическая обработка каталога: не зависит от исторических лайков.
    cleaned_texts = texts.where(texts.str.strip().ne(""), "emptytext")
    vectorizer = TfidfVectorizer(
        max_features=10_000, ngram_range=(1, 2), min_df=1,
        sublinear_tf=True, strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(cleaned_texts)
    if min(matrix.shape) >= 3:
        n_components = min(32, matrix.shape[0] - 1, matrix.shape[1] - 1)
        reduced = TruncatedSVD(
            n_components=n_components, n_iter=5, random_state=RANDOM_STATE
        ).fit_transform(matrix)
    else:
        reduced = matrix.toarray()
    reduced = normalize(reduced.astype(np.float32, copy=False))

    result["text_cluster"] = ""
    for topic, group in result.groupby("topic", sort=True):
        indices = group.index.to_numpy()
        # До 12 тематических кластеров; маленькие темы не дробим чрезмерно.
        n_clusters = min(12, max(1, len(indices) // 25))
        if n_clusters == 1:
            labels = np.zeros(len(indices), dtype=np.int32)
        else:
            clusterer = MiniBatchKMeans(
                n_clusters=n_clusters, random_state=RANDOM_STATE,
                n_init=3, batch_size=256,
            )
            labels = clusterer.fit_predict(reduced[indices])
        result.loc[indices, "text_cluster"] = [
            f"{topic}:c{int(label):02d}" for label in labels
        ]

    return cast_feature_types(
        result[["post_id", "topic", *POST_STATIC_COLUMNS]]
    )


def add_post_statistics(
    post_features: pd.DataFrame,
    feed_df: pd.DataFrame,
    post_smoothing: float = 20.0,
    topic_smoothing: float = 50.0,
) -> pd.DataFrame:
    """Считает статистику постов и тем по прошлой истории (не для Train)."""
    check_required_columns(post_features, ["post_id", "topic"], "post_features")
    check_required_columns(feed_df, ["post_id", "action"], "feed_df")
    views = feed_df.loc[feed_df.action == "view"]
    likes = feed_df.loc[feed_df.action == "like"]
    prior = len(likes) / len(views) if len(views) else 0.5

    post_views = views.groupby("post_id").size().rename("post_views")
    post_likes = likes.groupby("post_id").size().rename("post_likes")
    result = post_features.merge(post_views, on="post_id", how="left")
    result = result.merge(post_likes, on="post_id", how="left")
    for name in ["post_views", "post_likes"]:
        result[name] = result[name].fillna(0).astype(np.int32)
    result["post_ctr"] = (result.post_likes + post_smoothing * prior) / (
        result.post_views + post_smoothing
    )
    result["post_popularity"] = np.log1p(result.post_views)

    by_topic = feed_df[["post_id", "action"]].merge(
        post_features[["post_id", "topic"]], on="post_id", how="left", validate="many_to_one"
    )
    topic_views = by_topic.loc[by_topic.action == "view"].groupby("topic").size().rename("topic_views")
    topic_likes = by_topic.loc[by_topic.action == "like"].groupby("topic").size().rename("topic_likes")
    result = result.merge(topic_views, on="topic", how="left")
    result = result.merge(topic_likes, on="topic", how="left")
    for name in ["topic_views", "topic_likes"]:
        result[name] = result[name].fillna(0).astype(np.int32)
    result["topic_ctr"] = (result.topic_likes + topic_smoothing * prior) / (
        result.topic_views + topic_smoothing
    )
    by_cluster = feed_df[["post_id", "action"]].merge(
        post_features[["post_id", "text_cluster"]], on="post_id",
        how="left", validate="many_to_one",
    )
    cluster_views = by_cluster.loc[by_cluster.action == "view"].groupby(
        "text_cluster"
    ).size().rename("cluster_views")
    cluster_likes = by_cluster.loc[by_cluster.action == "like"].groupby(
        "text_cluster"
    ).size().rename("cluster_likes")
    cluster_stats = pd.concat([cluster_views, cluster_likes], axis=1).fillna(0)
    cluster_stats["cluster_ctr"] = (
        cluster_stats["cluster_likes"] + 30.0 * prior
    ) / (cluster_stats["cluster_views"] + 30.0)
    result["cluster_ctr"] = result["text_cluster"].map(
        cluster_stats["cluster_ctr"]
    ).fillna(prior)
    return cast_feature_types(result)


# ============================================================
# Признаки пользователей
# ============================================================

def build_user_features(
    users_df: pd.DataFrame,
    feed_df: pd.DataFrame,
    post_features: pd.DataFrame,
    positive_limit: int = 100,
    negative_limit: int = 100,
) -> pd.DataFrame:
    """Считает количество просмотров и лайков каждого пользователя."""
    check_required_columns(users_df, ["user_id", *USER_COLUMNS], "users_df")
    check_required_columns(feed_df, ["user_id", "post_id", "action"], "feed_df")
    activity = feed_df.groupby("user_id").agg(
        user_actions_count=("post_id", "size"),
        user_post_count=("post_id", "nunique"),
        user_likes_count=("action", lambda actions: (actions == "like").sum()),
        user_views_count=("action", lambda actions: (actions == "view").sum()),
    ).reset_index()
    result = users_df[["user_id", *USER_COLUMNS]].merge(
        activity, on="user_id", how="left", validate="one_to_one"
    )
    result[USER_HISTORY_COLUMNS] = result[USER_HISTORY_COLUMNS].fillna(0)
    return cast_feature_types(result)


# ============================================================
# Предпочтения пользователя по темам
# ============================================================

def build_user_topic_features(
    post_features: pd.DataFrame, feed_df: pd.DataFrame, smoothing: float = 10.0
) -> pd.DataFrame:
    """Считает просмотры, лайки и сглаженный CTR по темам пользователя."""
    check_required_columns(post_features, ["post_id", "topic", "topic_ctr"], "post_features")
    history = feed_df[["user_id", "post_id", "action"]].merge(
        post_features[["post_id", "topic", "topic_ctr"]], on="post_id", how="left",
        validate="many_to_one"
    )
    views = history.loc[history.action == "view"].groupby(["user_id", "topic"]).size().rename("user_topic_views")
    likes = history.loc[history.action == "like"].groupby(["user_id", "topic"]).size().rename("user_topic_likes")
    result = pd.concat([views, likes], axis=1).fillna(0).reset_index()
    if result.empty:
        return pd.DataFrame(columns=["user_id", "topic", *USER_TOPIC_COLUMNS])
    topic_prior = post_features.groupby("topic")["topic_ctr"].first()
    result["prior"] = result["topic"].map(topic_prior).fillna(0.5)
    result["user_topic_ctr"] = (
        result.user_topic_likes + smoothing * result.prior
    ) / (result.user_topic_views + smoothing)
    return cast_feature_types(result.drop(columns="prior"))


# ============================================================
# Исторические предпочтения пользователя по кластерам
# ============================================================

def build_user_cluster_features(
    post_features: pd.DataFrame,
    feed_df: pd.DataFrame,
    smoothing: float = 10.0,
) -> pd.DataFrame:
    """Считает просмотры, лайки и CTR по текстовым кластерам пользователя."""
    check_required_columns(
        post_features, ["post_id", "text_cluster", "cluster_ctr"], "post_features"
    )
    history = feed_df[["user_id", "post_id", "action"]].merge(
        post_features[["post_id", "text_cluster", "cluster_ctr"]],
        on="post_id", how="left", validate="many_to_one",
    )
    group_cols = ["user_id", "text_cluster"]
    views = history.loc[history.action == "view"].groupby(group_cols).size().rename(
        "user_cluster_views"
    )
    likes = history.loc[history.action == "like"].groupby(group_cols).size().rename(
        "user_cluster_likes"
    )
    result = pd.concat([views, likes], axis=1).fillna(0).reset_index()
    if result.empty:
        return pd.DataFrame(columns=["user_id", "text_cluster", *USER_CLUSTER_COLUMNS])
    prior = post_features.groupby("text_cluster")["cluster_ctr"].first()
    result["prior"] = result["text_cluster"].map(prior).fillna(0.5)
    result["user_cluster_ctr"] = (
        result.user_cluster_likes + smoothing * result.prior
    ) / (result.user_cluster_views + smoothing)
    return cast_feature_types(result.drop(columns="prior"))


# ============================================================
# История просмотренных постов
# ============================================================

def build_user_seen_features(feed_df: pd.DataFrame) -> pd.DataFrame:
    """Первое время просмотра каждой пары user_id + post_id для фильтрации в API."""
    check_required_columns(feed_df, ["user_id", "post_id", "timestamp", "action"], "feed_df")
    views = feed_df.loc[feed_df.action == "view", ["user_id", "post_id", "timestamp"]].copy()
    if views.empty:
        return pd.DataFrame(columns=["user_id", "seen_history_json"])
    views["timestamp"] = normalize_event_times(views["timestamp"])
    views = views.groupby(["user_id", "post_id"], as_index=False)["timestamp"].min()
    views = views.sort_values(["user_id", "timestamp", "post_id"])
    rows = []
    for user_id, group in views.groupby("user_id", sort=False):
        history = [
            [int(post_id), pd.Timestamp(timestamp).isoformat()]
            for post_id, timestamp in zip(group.post_id, group.timestamp)
        ]
        rows.append({"user_id": int(user_id), "seen_history_json": json.dumps(history)})
    return pd.DataFrame(rows, columns=["user_id", "seen_history_json"])


# ============================================================
# Сходство профилей с постом (необязательная совместимость)
# ============================================================

def add_similarity_features(dataframe: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Добавляет сходство эмбеддингов, если в таблице есть векторные признаки."""
    result = dataframe.copy() if copy else dataframe
    post_columns = sorted(
        [c for c in result if c.startswith("post_emb_")],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    if not post_columns:
        return result

    post_matrix = result[post_columns].to_numpy(dtype=np.float32)
    for user_prefix, out_prefix in [("user_pos_emb_", "positive_"), ("user_neg_emb_", "negative_")]:
        user_columns = sorted(
            [c for c in result if c.startswith(user_prefix)],
            key=lambda c: int(c.rsplit("_", 1)[1]),
        )
        if len(user_columns) != len(post_columns):
            raise ValueError(f"Не совпадают размеры эмбеддингов {user_prefix} и post_emb_")
        user_matrix = result[user_columns].to_numpy(dtype=np.float32)
        dot = (user_matrix * post_matrix).sum(axis=1)
        u_norm = np.linalg.norm(user_matrix, axis=1)
        p_norm = np.linalg.norm(post_matrix, axis=1)
        result[out_prefix + "dot"] = dot
        result[out_prefix + "cosine"] = dot / (u_norm * p_norm + 1e-8)
        result[out_prefix + "l2"] = np.linalg.norm(user_matrix - post_matrix, axis=1)
    result["preference_margin"] = result.positive_cosine - result.negative_cosine
    result["distance_margin"] = result.negative_l2 - result.positive_l2
    return result


# ============================================================
# Объединение таблиц признаков
# ============================================================

def build_dataset_from_features(
    events_df: pd.DataFrame,
    user_features: pd.DataFrame,
    post_features: pd.DataFrame,
    user_topic_features: pd.DataFrame,
    user_cluster_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Объединяет события с признаками из фиксированного среза истории."""
    times = build_time_features(events_df)
    result = times.merge(user_features, on="user_id", how="left", validate="many_to_one")
    result = result.merge(post_features, on="post_id", how="left", validate="many_to_one")
    if user_cluster_features is None:
        raise ValueError(
            "Для 32 признаков передайте user_cluster_features из snapshot"
        )
    result = result.merge(user_topic_features, on=["user_id", "topic"], how="left", validate="many_to_one")
    result = result.merge(
        user_cluster_features, on=["user_id", "text_cluster"],
        how="left", validate="many_to_one",
    )
    for column in ["user_topic_views", "user_topic_likes"]:
        result[column] = result[column].astype("float64").fillna(0)
    result["user_topic_ctr"] = result["user_topic_ctr"].astype("float32").fillna(
        result["topic_ctr"]
    )
    for column in ["user_cluster_views", "user_cluster_likes"]:
        result[column] = result[column].astype("float64").fillna(0)
    result["user_cluster_ctr"] = result["user_cluster_ctr"].astype("float32").fillna(
        result["cluster_ctr"]
    )
    return cast_feature_types(result[[*DROP_COLUMNS, *MODEL_FEATURES]])


def build_train_dataset(
    users_df: pd.DataFrame,
    posts_df: pd.DataFrame,
    feed_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    fit_text_pca: bool = False,
) -> pd.DataFrame:
    """Возвращает Train и Test вместе; не использовать для обучения модели."""
    if history_df is not None:
        raise ValueError("Передайте полный журнал через feed_df, отдельный history_df не нужен")
    posts = build_post_features(posts_df, fit_pca=fit_text_pca)
    train, test, _ = build_temporal_datasets(users_df, posts, feed_df)
    return pd.concat([train, test], ignore_index=True)


# ============================================================
# Исторические признаки: один проход по событиям (past-only)
# ============================================================

def prepare_temporal_feed(feed_df: pd.DataFrame) -> pd.DataFrame:
    """Подготовка и сортировка событий; target читается только у view."""
    columns = ["user_id", "post_id", "timestamp", "action", "target"]
    check_required_columns(feed_df, columns, "feed_df")
    feed = feed_df[columns].copy()
    feed["timestamp"] = normalize_event_times(feed["timestamp"])
    keys = ["user_id", "post_id", "timestamp", "action"]
    if feed[keys].isna().any().any() or feed.duplicated(keys).any():
        raise ValueError("В журнале есть пустые или повторяющиеся ключи событий")
    if not feed.action.isin(["view", "like"]).all():
        raise ValueError("Поддерживаются только view и like")
    view_mask = feed.action == "view"
    feed.loc[view_mask, "target"] = validated_target(feed.loc[view_mask, "target"])
    return feed.sort_values(keys[2:] + keys[:2], kind="stable").reset_index(drop=True)


def temporal_split_masks(feed_df: pd.DataFrame, split_date=SPLIT_DATE, require_both=True):
    """Train: view до/в границу; Test: view строго после границы.

    Для обычного Train/Test нужны обе части, а для snapshot — только история.
    """
    check_required_columns(feed_df, ["timestamp", "action"], "feed_df")
    times = normalize_event_times(feed_df["timestamp"])
    cutoff = pd.Timestamp(split_date)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)
    views = feed_df.action == "view"
    train_mask = views & (times <= cutoff)
    test_mask = views & (times > cutoff)
    if require_both and (not train_mask.any() or not test_mask.any()):
        raise ValueError(f"Нужны просмотры по обе стороны SPLIT_DATE={cutoff}")
    return train_mask, test_mask, cutoff


class _PastInteractions:
    """Только известные к текущему времени события. view.target не используется."""

    def __init__(self, users_df: pd.DataFrame, posts_df: pd.DataFrame, cold_start_ctr=0.5):
        check_required_columns(users_df, ["user_id", *USER_COLUMNS], "users_df")
        check_required_columns(
            posts_df, ["post_id", "topic", *POST_STATIC_COLUMNS], "posts_df"
        )
        if not 0 <= cold_start_ctr <= 1:
            raise ValueError("cold_start_ctr должен быть между 0 и 1")
        if users_df.user_id.isna().any() or users_df.user_id.duplicated().any():
            raise ValueError("Некорректные user_id")
        if posts_df.post_id.isna().any() or posts_df.post_id.duplicated().any():
            raise ValueError("Некорректные post_id")

        self.static_users = users_df[["user_id", *USER_COLUMNS]].copy()
        self.static_posts = posts_df[["post_id", "topic", *POST_STATIC_COLUMNS]].copy()
        for name in ["country", "city", "os", "source"]:
            self.static_users[name] = self.static_users[name].fillna("unknown").astype(str)
        self.static_posts["topic"] = self.static_posts.topic.fillna("unknown").astype(str)
        self.static_posts["text_cluster"] = self.static_posts.text_cluster.astype(str)

        self.users = self.static_users.set_index("user_id").to_dict("index")
        self.posts = self.static_posts.set_index("post_id").to_dict("index")
        self.activity = defaultdict(lambda: [0, set(), 0, 0])  # actions, unique posts, likes, views
        self.post_counts = defaultdict(lambda: [0, 0])       # views, likes
        self.topic_counts = defaultdict(lambda: [0, 0])
        self.user_topic_counts = defaultdict(lambda: [0, 0])
        self.cluster_counts = defaultdict(lambda: [0, 0])
        self.user_cluster_counts = defaultdict(lambda: [0, 0])
        self.total_views = 0
        self.total_likes = 0
        self.cold_start_ctr = float(cold_start_ctr)

    def global_ctr(self) -> float:
        """Возвращает долю лайков в истории с учётом холодного старта."""
        return self.total_likes / self.total_views if self.total_views else self.cold_start_ctr

    def user_row(self, user_id: int) -> dict:
        """Возвращает текущие счётчики и признаки пользователя."""
        actions, unique_posts, likes, views = self.activity[user_id]
        return {
            **self.users[user_id],
            "user_actions_count": actions,
            "user_post_count": len(unique_posts),
            "user_likes_count": likes,
            "user_views_count": views,
        }

    def post_row(self, post_id: int) -> dict:
        """Возвращает статистику поста до текущего события."""
        topic = self.posts[post_id]["topic"]
        pv, pl = self.post_counts[post_id]
        tv, tl = self.topic_counts[topic]
        prior = self.global_ctr()
        cluster = self.posts[post_id]["text_cluster"]
        cv, cl = self.cluster_counts[cluster]
        return {
            "topic": topic,
            "text_cluster": cluster,
            "text_length": self.posts[post_id]["text_length"],
            "word_count": self.posts[post_id]["word_count"],
            "cluster_ctr": (cl + 30 * prior) / (cv + 30),
            "post_views": pv,
            "post_likes": pl,
            "post_ctr": (pl + 20 * prior) / (pv + 20),
            "post_popularity": np.log1p(pv),
            "topic_views": tv,
            "topic_likes": tl,
            "topic_ctr": (tl + 50 * prior) / (tv + 50),
        }

    def topic_row(self, user_id: int, topic: str) -> dict:
        """Возвращает статистику пользователя по теме."""
        uv, ul = self.user_topic_counts[(user_id, topic)]
        tv, tl = self.topic_counts[topic]
        topic_ctr = (tl + 50 * self.global_ctr()) / (tv + 50)
        return {
            "user_topic_views": uv,
            "user_topic_likes": ul,
            "user_topic_ctr": (ul + 10 * topic_ctr) / (uv + 10),
        }

    def cluster_row(self, user_id: int, cluster: str) -> dict:
        """Возвращает статистику пользователя по текстовому кластеру."""
        uv, ul = self.user_cluster_counts[(user_id, cluster)]
        cv, cl = self.cluster_counts[cluster]
        prior = (cl + 30 * self.global_ctr()) / (cv + 30)
        return {
            "user_cluster_views": uv,
            "user_cluster_likes": ul,
            "user_cluster_ctr": (ul + 10 * prior) / (uv + 10),
        }

    def observe(self, user_id: int, post_id: int, action: str) -> None:
        """Обновляет счётчики после обработки события."""
        activity = self.activity[user_id]
        activity[0] += 1
        activity[1].add(post_id)
        topic = self.posts[post_id]["topic"]
        cluster = self.posts[post_id]["text_cluster"]
        if action == "view":
            activity[3] += 1
            self.total_views += 1
            self.post_counts[post_id][0] += 1
            self.topic_counts[topic][0] += 1
            self.user_topic_counts[(user_id, topic)][0] += 1
            self.cluster_counts[cluster][0] += 1
            self.user_cluster_counts[(user_id, cluster)][0] += 1
        else:  # 'like': known only at the timestamp of the like event
            activity[2] += 1
            self.total_likes += 1
            self.post_counts[post_id][1] += 1
            self.topic_counts[topic][1] += 1
            self.user_topic_counts[(user_id, topic)][1] += 1
            self.cluster_counts[cluster][1] += 1
            self.user_cluster_counts[(user_id, cluster)][1] += 1

    def tables(self) -> dict[str, pd.DataFrame]:
        """Собирает четыре таблицы исторических признаков."""
        users = pd.DataFrame(
            [{"user_id": uid, **self.user_row(uid)} for uid in self.users],
            columns=["user_id", *USER_COLUMNS, *USER_HISTORY_COLUMNS],
        )
        posts = pd.DataFrame(
            [{"post_id": pid, **self.post_row(pid)} for pid in self.posts],
            columns=["post_id", "topic", *POST_STATIC_COLUMNS,
                     *POST_HISTORY_COLUMNS, *POST_EXTRA_COLUMNS],
        )
        topics = pd.DataFrame(
            [
                {"user_id": uid, "topic": topic, **self.topic_row(uid, topic)}
                for uid, topic in self.user_topic_counts
            ],
            columns=["user_id", "topic", *USER_TOPIC_COLUMNS],
        )
        clusters = pd.DataFrame(
            [
                {"user_id": uid, "text_cluster": cluster,
                 **self.cluster_row(uid, cluster)}
                for uid, cluster in self.user_cluster_counts
            ],
            columns=["user_id", "text_cluster", *USER_CLUSTER_COLUMNS],
        )
        return {
            "user_features": cast_feature_types(users),
            "post_features": cast_feature_types(posts),
            "user_topic_features": cast_feature_types(topics),
            "user_cluster_features": cast_feature_types(clusters),
        }


def _replay_history(
    users_df: pd.DataFrame,
    post_features: pd.DataFrame,
    history: pd.DataFrame,
    cold_start_ctr: float = 0.5,
    emit_views: bool = True,
):
    """Показатели view вычисляются до применения событий с тем же timestamp."""
    state = _PastInteractions(users_df, post_features, cold_start_ctr)
    if not set(history.user_id).issubset(state.users) or not set(history.post_id).issubset(state.posts):
        raise ValueError("В feed есть пользователи или посты вне справочников")

    columns = [*DROP_COLUMNS, *MODEL_FEATURES]
    n = int((history.action == "view").sum()) if emit_views else 0
    arrays = {}
    for col in columns:
        if col == "timestamp":
            dtype = "datetime64[ns]"
        elif col in {"user_id", "post_id"}:
            dtype = np.int64
        elif col in CATEGORICAL_COLUMNS:
            dtype = object
        elif col in FLOAT_COLUMNS or col in {"gender", "age", "exp_group"}:
            dtype = np.float32
        elif col in {"target", "is_weekend"}:
            dtype = np.int8
        else:
            dtype = np.int32
        arrays[col] = np.empty(n, dtype=dtype)

    idx = 0
    # Равные timestamp обрабатываются одним пакетом, чтобы исключить утечку
    # между событиями с одной и той же отметкой времени.
    for timestamp, events_it in groupby(history.itertuples(index=False), key=lambda row: row.timestamp):
        events = list(events_it)
        if emit_views:
            for event in events:
                if event.action != "view":
                    continue
                post = state.post_row(event.post_id)
                row = {
                    "user_id": event.user_id,
                    "post_id": event.post_id,
                    "timestamp": timestamp,
                    "target": int(event.target),
                    "hour": timestamp.hour,
                    "day_of_week": timestamp.dayofweek,
                    "month": timestamp.month,
                    "is_weekend": int(timestamp.dayofweek >= 5),
                    **state.user_row(event.user_id),
                    **post,
                    **state.topic_row(event.user_id, post["topic"]),
                    **state.cluster_row(event.user_id, post["text_cluster"]),
                }
                for col in columns:
                    arrays[col][idx] = row[col]
                idx += 1
        for event in events:
            state.observe(event.user_id, event.post_id, event.action)

    if not emit_views:
        return pd.DataFrame(columns=columns), state
    return cast_feature_types(pd.DataFrame(arrays, copy=False), copy=False), state


def _history_tables(state: _PastInteractions, history: pd.DataFrame, cutoff: pd.Timestamp):
    """Четыре таблицы признаков и одна таблица просмотренных постов."""
    result = state.tables()
    result["user_seen_features"] = build_user_seen_features(history)
    for table in result.values():
        table.attrs["feature_definition"] = FEATURE_DEFINITION
        table.attrs["history_cutoff"] = pd.Timestamp(cutoff).isoformat()
        table.attrs["cold_start_ctr"] = state.cold_start_ctr
    return result


def build_history_snapshot(
    users_df: pd.DataFrame,
    base_post_features: pd.DataFrame,
    feed_df: pd.DataFrame,
    split_date=SPLIT_DATE,
    cold_start_ctr: float = 0.5,
):
    """Строит пять таблиц признаков по истории до указанной даты."""
    feed = prepare_temporal_feed(feed_df)
    _, _, cutoff = temporal_split_masks(feed, split_date, require_both=False)
    history = feed.loc[feed.timestamp <= cutoff].reset_index(drop=True)
    _, state = _replay_history(
        users_df, base_post_features, history,
        cold_start_ctr=cold_start_ctr, emit_views=False,
    )
    return _history_tables(state, history, cutoff)


def build_temporal_datasets(
    users_df: pd.DataFrame,
    base_post_features: pd.DataFrame,
    feed_df: pd.DataFrame,
    split_date=SPLIT_DATE,
    cold_start_ctr: float = 0.5,
):
    """Train — строго past-only; Test — фиксированные таблицы на границе."""
    feed = prepare_temporal_feed(feed_df)
    _, test_mask, cutoff = temporal_split_masks(feed, split_date)
    history = feed.loc[feed.timestamp <= cutoff].reset_index(drop=True)
    train, state = _replay_history(users_df, base_post_features, history, cold_start_ctr)
    tables = _history_tables(state, history, cutoff)
    test = build_dataset_from_features(
        feed.loc[test_mask], tables["user_features"], tables["post_features"],
        tables["user_topic_features"], tables["user_cluster_features"],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True), tables


# ============================================================
# Оценка, экспорт таблиц и совместимость модели с API
# ============================================================

def print_classification_metrics(model, train: pd.DataFrame, test: pd.DataFrame) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    for name, data in [("Train", train), ("Test", test)]:
        y = data["target"].astype(int)
        X = data.loc[:, list(model.feature_names_in_)]
        scores = model.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, scores) if y.nunique() == 2 else float("nan")
        ap = average_precision_score(y, scores) if y.sum() else float("nan")
        print(f"{name}: rows={len(data)}, positive_share={y.mean():.6f}")
        print(f"{name} ROC-AUC: {roc_auc:.6f}")
        print(f"{name} Average Precision: {ap:.6f}")


def mark_temporal_model(model, snapshot: dict):
    """Сохраняет версию признаков и дату истории в объекте модели."""
    metadata = snapshot["post_features"].attrs
    model.feature_definition_ = metadata["feature_definition"]
    model.history_cutoff_ = metadata["history_cutoff"]
    model.cold_start_ctr_ = metadata["cold_start_ctr"]
    return model


def snapshot_export_tables(snapshot: dict) -> dict[str, pd.DataFrame]:
    """Экспорт пяти таблиц истории для инференса с 32 признаками."""
    result = {}
    for name in [
        "user_features", "post_features", "user_topic_features",
        "user_cluster_features", "user_seen_features",
    ]:
        table = snapshot[name]
        export = table.copy()
        export["_feature_definition"] = table.attrs["feature_definition"]
        export["_history_cutoff"] = table.attrs["history_cutoff"]
        result[name] = export
    return result


def validate_model_snapshot(models, tables: dict) -> dict[str, pd.DataFrame]:
    """Проверяет версию признаков и дату среза моделей и таблиц."""
    if "user_cluster_features" not in tables:
        raise ValueError("Для текстовых признаков отсутствует user_cluster_features")
    model_definitions = [getattr(model, "feature_definition_", None) for model in models]
    # Не разрешаем старые таблицы без метаданных даже со старой моделью.
    if any(definition != FEATURE_DEFINITION for definition in model_definitions):
        raise ValueError("Нельзя смешивать старые модели с новыми таблицами признаков")
    cutoffs = {str(model.history_cutoff_) for model in models}
    if len(cutoffs) != 1:
        raise ValueError("У моделей разные границы истории")
    cutoff = cutoffs.pop()
    clean = {}
    for name, table in tables.items():
        if not set(TABLE_METADATA).issubset(table.columns):
            raise ValueError(f"У таблицы {name} отсутствует дата среза")
        if not table.empty:
            if table["_feature_definition"].nunique(dropna=False) != 1 or (
                table["_feature_definition"].iloc[0] != FEATURE_DEFINITION
            ):
                raise ValueError(f"Неверный формат таблицы {name}")
            if table["_history_cutoff"].nunique(dropna=False) != 1 or (
                table["_history_cutoff"].iloc[0] != cutoff
            ):
                raise ValueError(f"Модель и таблица {name} имеют разные даты среза")
        clean[name] = table.drop(columns=TABLE_METADATA)
    return clean
