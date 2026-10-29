"""Нейросетевые признаки постов: SentenceTransformer -> PCA -> MiniBatchKMeans.

Признаки вычисляются только по текстам постов, без target и будущих лайков.
Кэш привязан к содержимому каталога, версии модели и настройкам обработки.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

DEFAULT_TRANSFORMER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
N_CLUSTERS = 16
SEMANTIC_COLUMNS = [f"sem_dist_{i:02d}" for i in range(N_CLUSTERS)]


def _catalog(posts: pd.DataFrame) -> pd.DataFrame:
    """Проверяет каталог постов и упорядочивает его по post_id."""
    data = posts.rename(columns={"id": "post_id"}) if "post_id" not in posts.columns else posts
    needed = {"post_id", "text"}
    if not needed.issubset(data.columns):
        raise ValueError(f"Не хватает колонок постов: {sorted(needed - set(data.columns))}")
    if data.post_id.isna().any() or data.post_id.duplicated().any():
        raise ValueError("post_id должен быть уникальным и непустым")
    result = data[["post_id", "text"]].copy().sort_values("post_id", kind="stable")
    result["post_id"] = result["post_id"].astype(np.int64)
    result["text"] = result["text"].fillna("").astype(str)
    return result.reset_index(drop=True)


def _fingerprint(posts: pd.DataFrame, model_name: str) -> str:
    """Хеш каталога и настроек для проверки кэша расстояний."""
    sha = hashlib.sha256()
    for value in (model_name, f"clusters={N_CLUSTERS}", "pca=50", "seed=42"):
        sha.update(value.encode("utf-8"))
        sha.update(b"\0")
    for post_id, value in zip(posts["post_id"], posts["text"]):
        sha.update(str(int(post_id)).encode("ascii"))
        sha.update(b"\0")
        sha.update(value.encode("utf-8"))
        sha.update(b"\0")
    return sha.hexdigest()


def distances_from_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Сокращает размерность и возвращает 16 расстояний до центров кластеров."""
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < N_CLUSTERS or array.shape[1] < 2:
        raise ValueError(f"Нужно минимум {N_CLUSTERS} постов и 2 координаты эмбеддинга")
    if not np.isfinite(array).all():
        raise ValueError("Эмбеддинги содержат NaN или inf")

    # PCA обучается на текстах постов, без пользовательских меток.
    components = min(50, array.shape[0] - 1, array.shape[1])
    if components < array.shape[1]:
        reduced = PCA(
            n_components=components, svd_solver="randomized", random_state=42
        ).fit_transform(array)
    else:
        reduced = array
    reduced = normalize(reduced).astype(np.float32, copy=False)

    clusterer = MiniBatchKMeans(
        n_clusters=N_CLUSTERS, n_init=3, batch_size=256, random_state=42
    )
    clusterer.fit(reduced)
    return clusterer.transform(reduced).astype(np.float32)


def build_semantic_post_features(
    posts: pd.DataFrame,
    cache_path: str | Path = "semantic_post_features.npz",
    model_name: str = DEFAULT_TRANSFORMER,
    device: str = "cpu",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Загружает или вычисляет расстояния до 16 текстовых кластеров."""
    if batch_size < 1:
        raise ValueError("batch_size должен быть положительным")
    data = _catalog(posts)
    if len(data) < N_CLUSTERS:
        raise ValueError(f"В каталоге должно быть хотя бы {N_CLUSTERS} постов")
    digest = _fingerprint(data, model_name)
    cache_path = Path(cache_path)
    ids = data["post_id"].to_numpy(dtype=np.int64)
    values = None

    if cache_path.is_file():
        # NPZ загружаем без pickle; кэш не должен содержать исполняемые объекты.
        with np.load(cache_path, allow_pickle=False) as cached:
            if (
                cached["fingerprint"].item() == digest
                and np.array_equal(cached["post_ids"], ids)
                and cached["distances"].shape == (len(data), N_CLUSTERS)
            ):
                values = cached["distances"].astype(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError("Кэш содержит некорректные расстояния")
                print(f"Семантические признаки загружены из {cache_path}", flush=True)
            else:
                raise ValueError(
                    f"Несовместимый кэш {cache_path}: изменился каталог или модель. "
                    "Укажите новый --semantic-cache, чтобы сохранить предыдущий эксперимент."
                )

    if values is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Установите зависимость: python -m pip install sentence-transformers"
            ) from exc
        print(f"Загружаем Transformer {model_name} на устройстве {device}...", flush=True)
        encoder = SentenceTransformer(model_name, device=device)
        embeddings = encoder.encode(
            data["text"].tolist(), batch_size=batch_size,
            normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=True,
        )
        values = distances_from_embeddings(embeddings)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, fingerprint=np.array(digest), post_ids=ids, distances=values
        )
        print(f"Семантические признаки сохранены в {cache_path}", flush=True)

    frame = pd.DataFrame(values, columns=SEMANTIC_COLUMNS)
    frame.insert(0, "post_id", ids)
    return frame


def attach_semantic_features(events: pd.DataFrame, semantic_posts: pd.DataFrame) -> pd.DataFrame:
    """Добавляет семантические признаки по post_id без изменения числа строк."""
    if not events["post_id"].notna().all():
        raise ValueError("В событиях отсутствуют post_id")
    if semantic_posts["post_id"].duplicated().any():
        raise ValueError("В семантических признаках повторяются post_id")
    if any(column in events.columns for column in SEMANTIC_COLUMNS):
        raise ValueError("Семантические признаки уже присутствуют в событиях")

    enriched = events.merge(
        semantic_posts[["post_id", *SEMANTIC_COLUMNS]],
        on="post_id", how="left", sort=False, validate="many_to_one",
    )
    if len(enriched) != len(events) or enriched[SEMANTIC_COLUMNS].isna().any().any():
        raise ValueError("Не для каждого post_id найдены семантические признаки")
    return enriched
