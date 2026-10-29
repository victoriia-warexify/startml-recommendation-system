"""Косинусное сходство поста с профилем лайков пользователя.

Профиль строится только по уникальным лайкам из доступной истории.
Для Train используется прошлое каждого события, для Validation/Test — срез на дату.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from semantic_post_features import DEFAULT_TRANSFORMER, _catalog

PROFILE_COLUMN = "user_post_similarity"
EMBED_DIM = 50


def _fingerprint(posts: pd.DataFrame, model_name: str) -> str:
    """Хеш каталога и настроек для проверки кэша векторов."""
    digest = hashlib.sha256()
    for token in (model_name, "profile_pca=50", "profile_seed=42", "profile_normalized_v1"):
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
    for pid, txt in zip(posts.post_id, posts.text):
        digest.update(str(int(pid)).encode("ascii"))
        digest.update(b"\0")
        digest.update(txt.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _reduce_vectors(embeddings: np.ndarray) -> np.ndarray:
    """Сокращает эмбеддинги через PCA и нормирует их."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] < 3 or embeddings.shape[1] < 2:
        raise ValueError("Для построения профилей нужны минимум три двумерных эмбеддинга")
    if not np.isfinite(embeddings).all():
        raise ValueError("Эмбеддинги содержат NaN/inf")
    components = min(EMBED_DIM, embeddings.shape[0] - 1, embeddings.shape[1])
    if components < embeddings.shape[1]:
        embeddings = PCA(
            n_components=components, svd_solver="randomized", random_state=42
        ).fit_transform(embeddings)
    return normalize(embeddings).astype(np.float32)


def build_profile_post_vectors(
    posts: pd.DataFrame,
    cache_path: Path | str,
    model_name: str = DEFAULT_TRANSFORMER,
    device: str = "cpu",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Загружает или вычисляет нормированные векторы постов для профилей."""
    if batch_size < 1:
        raise ValueError("batch_size должен быть > 0")
    catalog = _catalog(posts)
    fingerprint = _fingerprint(catalog, model_name)
    ids = catalog.post_id.to_numpy(dtype=np.int64)
    path = Path(cache_path)
    if path.exists():
        with np.load(path, allow_pickle=False) as blob:
            if (
                blob["fingerprint"].item() != fingerprint
                or not np.array_equal(blob["post_ids"], ids)
            ):
                raise ValueError("Кэш векторов не совпадает с каталогом/Transformer; задайте другой --profile-cache")
            vectors = blob["vectors"].astype(np.float32)
            if vectors.shape[0] != len(ids) or vectors.shape[1] < 2 or not np.isfinite(vectors).all():
                raise ValueError("Кэш векторов повреждён")
        print(f"Загружены векторы интересов из {path}", flush=True)
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Установите sentence-transformers") from exc
        encoder = SentenceTransformer(model_name, device=device)
        raw = encoder.encode(
            catalog.text.tolist(), batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True,
        )
        vectors = _reduce_vectors(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, fingerprint=np.array(fingerprint), post_ids=ids, vectors=vectors)
        print(f"Векторы для профилей сохранены: {path}", flush=True)
    # Векторы нужны только для эксперимента и не выгружаются в PostgreSQL.
    return pd.DataFrame({"post_id": ids, "vector": list(vectors)})


def add_user_post_similarity(
    events: pd.DataFrame,
    feed: pd.DataFrame,
    vector_posts: pd.DataFrame,
    *,
    mode: str,
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Добавляет сходство с профилем из прошлых лайков без использования target.

    past — лайки строго до события; fixed — лайки до указанной даты включительно.
    """
    if mode not in ("past", "fixed"):
        raise ValueError("mode должен быть past/fixed")
    if mode == "fixed" and cutoff is None:
        raise ValueError("Для mode=fixed нужна граница cutoff")
    if mode == "fixed":
        cutoff = pd.Timestamp(cutoff)
        if pd.isna(cutoff):
            raise ValueError("cutoff не должен быть NaT")
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
    cols = {"user_id", "post_id", "timestamp"}
    if not cols.issubset(events.columns) or not cols.union({"action"}).issubset(feed.columns):
        raise ValueError("Нужны user_id/post_id/timestamp и action в журнале")
    if PROFILE_COLUMN in events.columns:
        raise ValueError("Признак сходства уже присутствует")
    if vector_posts.post_id.duplicated().any() or vector_posts.post_id.isna().any():
        raise ValueError("Некорректный каталог эмбеддингов")
    if vector_posts.empty:
        raise ValueError("Пустой каталог эмбеддингов")

    matrix = np.stack(vector_posts.vector.to_numpy()).astype(np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Некорректные векторы постов")
    id_to_ix = pd.Series(np.arange(len(vector_posts)), index=vector_posts.post_id)
    post_ix = id_to_ix.reindex(events.post_id).to_numpy()
    if pd.isna(post_ix).any():
        raise ValueError("Для части кандидатов нет текстовых векторов")
    post_ix = post_ix.astype(np.int64)

    likes = feed.loc[feed.action == "like", ["user_id", "post_id", "timestamp"]].copy()
    likes["timestamp"] = pd.to_datetime(likes.timestamp, utc=True).dt.tz_localize(None)
    if mode == "fixed":
        likes = likes.loc[likes.timestamp <= cutoff]
    # Один и тот же пост учитываем в профиле один раз.
    likes = likes.sort_values(["timestamp", "user_id", "post_id"], kind="stable")
    likes = likes.drop_duplicates(["user_id", "post_id"], keep="first")
    like_ix = id_to_ix.reindex(likes.post_id).to_numpy()
    if pd.isna(like_ix).any():
        raise ValueError("Есть лайки постов, которых нет в каталоге")
    likes["_vector_ix"] = like_ix.astype(np.int64)

    timestamps = pd.to_datetime(events.timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    if pd.isna(timestamps).any():
        raise ValueError("timestamp событий не должен быть пустым")
    if mode == "fixed" and (timestamps <= np.datetime64(cutoff)).any():
        raise ValueError("Для fixed все события должны быть позже cutoff")
    score = np.zeros(len(events), dtype=np.float32)
    likes_by_user = {user: grp for user, grp in likes.groupby("user_id", sort=False)}
    # groupby.indices возвращает позиции строк, даже если индекс не последовательный.
    for user, positions in events.groupby("user_id", sort=False).indices.items():
        grouped = likes_by_user.get(user)
        if grouped is None or grouped.empty:
            continue
        indices = np.asarray(positions, dtype=np.int64)
        liked_vecs = matrix[grouped._vector_ix.to_numpy(dtype=np.int64)]
        if mode == "fixed":
            profile = liked_vecs.mean(axis=0)
            length = np.linalg.norm(profile)
            if length > 1e-12:
                score[indices] = (matrix[post_ix[indices]] @ (profile / length)).astype(np.float32)
        else:
            # side='left' исключает лайки с тем же timestamp, что и просмотр.
            like_times = grouped.timestamp.to_numpy(dtype="datetime64[ns]")
            cumulatives = np.cumsum(liked_vecs, axis=0, dtype=np.float32)
            n_prior = np.searchsorted(like_times, timestamps[indices], side="left")
            valid = n_prior > 0
            if valid.any():
                v_ix = indices[valid]
                profiles = cumulatives[n_prior[valid] - 1]
                norms = np.linalg.norm(profiles, axis=1)
                safe = norms > 1e-12
                values = np.zeros(len(v_ix), dtype=np.float32)
                values[safe] = np.einsum(
                    "ij,ij->i", profiles[safe] / norms[safe, None], matrix[post_ix[v_ix[safe]]]
                )
                score[v_ix] = values
    enriched = events.copy()
    enriched[PROFILE_COLUMN] = np.clip(score, -1, 1)
    return enriched
