"""Эксперимент: исходная MLP + 16 расстояний от Transformer-эмбеддингов.

Не меняет model_test.py, features.py, HGB, API и старые модели.
Отбор числа эпох — по внутренней временной Validation (28.11–12.12).
Test используется только для итогового диагностического сравнения.

Пример:
  python train_mlp_semantic.py --user-mod 20 \\
    --baseline-model models_5pct/model_test.pkl \\
    --control-model models_5pct/model_control.pkl \\
    --semantic-cache cache/post_semantic_distances.npz \\
    --output-model models_semantic_5pct/model_test.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from connection import load_data
from features import (
    MODEL_FEATURES, SPLIT_DATE, build_post_features,
    build_temporal_datasets, build_history_snapshot,
    build_dataset_from_features, prepare_temporal_feed, mark_temporal_model,
)
from model_test import (
    INNER_SPLIT_DATE, create_model, sample_training_rows,
)
from semantic_post_features import (
    DEFAULT_TRANSFORMER, SEMANTIC_COLUMNS, attach_semantic_features,
    build_semantic_post_features,
)
from evaluate_observed_hitrate import audit_unique_posts, probabilities_in_batches

FEATURES = [*MODEL_FEATURES, *SEMANTIC_COLUMNS]


def metrics(y: pd.Series, prediction: np.ndarray) -> dict:
    """Вычисляет ROC-AUC и Average Precision."""
    labels = y.astype(np.int8)
    return {
        "roc_auc": float(roc_auc_score(labels, prediction)),
        "ap": float(average_precision_score(labels, prediction)),
    }


def train_epochs(train: pd.DataFrame, valid: pd.DataFrame, max_epochs: int, patience: int):
    """Выбирает число эпох по Average Precision на временной Validation."""
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs и patience должны быть положительными")
    fit_rows = sample_training_rows(train)
    X_fit = fit_rows[FEATURES]
    y_fit = fit_rows.target.astype(np.int8)
    X_val = valid[FEATURES]
    y_val = valid.target.astype(np.int8)
    if y_fit.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError("Оба класса должны быть представлены в Train и Validation")

    pipe = create_model(X_fit)
    processor = pipe.named_steps["preprocessor"]
    classifier = pipe.named_steps["classifier"]
    X_encoded = np.asarray(processor.fit_transform(X_fit, y_fit), dtype=np.float32)
    V_encoded = np.asarray(processor.transform(X_val), dtype=np.float32)
    best_ap = -1.0
    best_epoch = 1
    bad_epochs = 0
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            classifier.partial_fit(X_encoded, y_fit, classes=np.array([0, 1]))
        else:
            classifier.partial_fit(X_encoded, y_fit)
        values = classifier.predict_proba(V_encoded)[:, 1]
        ap = float(average_precision_score(y_val, values))
        print(f"  Эпоха {epoch:02d}: Validation AP={ap:.6f}", flush=True)
        if ap > best_ap + 1e-4:
            best_ap, best_epoch, bad_epochs = ap, epoch, 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    return best_epoch, best_ap


def fit_final(train: pd.DataFrame, epochs: int):
    """Обучает финальную MLP на полном Train за выбранное число эпох."""
    if epochs < 1:
        raise ValueError("epochs должен быть положительным")
    rows = sample_training_rows(train)
    X, y = rows[FEATURES], rows.target.astype(np.int8)
    pipe = create_model(X)
    processor = pipe.named_steps["preprocessor"]
    classifier = pipe.named_steps["classifier"]
    X_encoded = np.asarray(processor.fit_transform(X, y), dtype=np.float32)
    for epoch in range(epochs):
        if epoch == 0:
            classifier.partial_fit(X_encoded, y, classes=np.array([0, 1]))
        else:
            classifier.partial_fit(X_encoded, y)
    return pipe


def load_trusted_model(path: str | None):
    """Загружает локальную модель для сравнения результатов."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Нет модели: {path}")
    with path.open("rb") as file:
        return pickle.load(file)  # Только локальные доверенные pkl.


def main():
    """Запускает семантический эксперимент без изменения основных моделей."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-mod", type=int, default=20)
    parser.add_argument("--semantic-cache", type=Path, default=Path("cache/post_semantic_distances.npz"))
    parser.add_argument("--transformer", default=DEFAULT_TRANSFORMER)
    parser.add_argument("--device", default="cpu", help="cpu или mps (для поддерживаемых Mac)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch для Transformer")
    parser.add_argument("--score-batch-size", type=int, default=20000, help="Batch для predict_proba")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--baseline-model", default="models_5pct/model_test.pkl")
    parser.add_argument("--control-model", default="models_5pct/model_control.pkl")
    parser.add_argument("--output-model", type=Path, default=Path("models_semantic_5pct/model_test.pkl"))
    args = parser.parse_args()
    if min(args.user_mod, args.batch_size, args.score_batch_size, args.max_epochs, args.patience) < 1:
        parser.error("Числовые параметры должны быть положительными")
    if args.output_model.exists():
        parser.error(f"Модель уже существует: {args.output_model} (перезапись запрещена)")

    print(f"Загружаем когорту user_id % {args.user_mod} = 0...", flush=True)
    users, posts, feed = load_data(user_mod=args.user_mod)
    semantic_posts = build_semantic_post_features(
        posts, args.semantic_cache, args.transformer, args.device, args.batch_size,
    )
    print("Строим прежние 32 признака и временные Train/Test...", flush=True)
    static_posts = build_post_features(posts, fit_pca=False)
    train, test, snapshot = build_temporal_datasets(users, static_posts, feed)

    # Семантические признаки берутся только из текстов; это merge по post_id.
    train = attach_semantic_features(train, semantic_posts)
    test = attach_semantic_features(test, semantic_posts)
    print(f"Train: {len(train):,}; Test: {len(test):,}; признаков: {len(FEATURES)}", flush=True)

    normalized_feed = prepare_temporal_feed(feed)
    # Внутренняя Validation не должна использовать историю после 28 ноября.
    inner_snapshot = build_history_snapshot(
        users, static_posts, normalized_feed, split_date=INNER_SPLIT_DATE
    )
    val_events = normalized_feed.loc[
        (normalized_feed.timestamp > INNER_SPLIT_DATE)
        & (normalized_feed.timestamp <= SPLIT_DATE)
    ]
    val = build_dataset_from_features(
        val_events, inner_snapshot["user_features"], inner_snapshot["post_features"],
        inner_snapshot["user_topic_features"], inner_snapshot["user_cluster_features"],
    )
    val = attach_semantic_features(val, semantic_posts)
    inner_train = train.loc[train.timestamp <= INNER_SPLIT_DATE]
    print(f"Внутренний Train: {len(inner_train):,}; Validation: {len(val):,}", flush=True)

    best_epoch, val_ap = train_epochs(inner_train, val, args.max_epochs, args.patience)
    print(f"Лучшая эпоха={best_epoch}, Validation AP={val_ap:.6f}", flush=True)
    del inner_snapshot, val, inner_train

    print("Обучаем итоговую экспериментальную MLP на полном Train...", flush=True)
    model = fit_final(train, best_epoch)
    mark_temporal_model(model, snapshot)
    model.feature_definition_ = f"{model.feature_definition_}+semantic_dist16_v1"
    model.semantic_model_ = args.transformer
    model.semantic_feature_columns_ = list(SEMANTIC_COLUMNS)

    print("Оцениваем на прежнем Test без изменения состава событий...", flush=True)
    scores = {
        "MLP_semantic": probabilities_in_batches(model, test, FEATURES, args.score_batch_size)
    }
    print("MLP_semantic Test:", metrics(test.target, scores["MLP_semantic"]), flush=True)
    for name, path in (("MLP_baseline", args.baseline_model), ("HGB", args.control_model)):
        baseline = load_trusted_model(path)
        if baseline is not None:
            names = list(baseline.feature_names_in_)
            score = probabilities_in_batches(baseline, test, names, args.score_batch_size)
            scores[name] = score
            print(f"{name} Test:", metrics(test.target, score), flush=True)
    scores["popularity"] = test.post_likes.to_numpy(dtype=np.float64)

    summary, _ = audit_unique_posts(test, scores, k=5)
    print("\nObserved HitRate@5: уникальные посты, target первого просмотра", flush=True)
    for name, score in summary["first_hitrate"].items():
        print(f"  {name:16s}: {score:.6f}", flush=True)
    print("Диагностическая offline-метрика по просмотренным постам, не LMS и не онлайн A/B.")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with args.output_model.open("xb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Экспериментальная модель сохранена: {args.output_model}", flush=True)
    print("Не подключайте эту модель к существующему app.py без обновления snapshot таблиц.")


if __name__ == "__main__":
    main()
