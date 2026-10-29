"""Экспериментальная MLP с профилем интересов по уникальным лайкам.

Train: лайки до события. Validation: срез на 28.11.2021.
Test: срез на 12.12.2021. Число эпох выбирается только по Validation.
Скрипт не изменяет базовые модели и таблицы PostgreSQL.

Пример:
  python train_mlp_personalized.py --user-mod 20 \
    --profile-cache cache/post_profile_vectors.npz \
    --output-dir models_personalized_5pct
"""
from __future__ import annotations

import argparse
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from connection import load_data
from features import (
    MODEL_FEATURES, SPLIT_DATE, build_post_features, build_temporal_datasets,
    build_history_snapshot, build_dataset_from_features, prepare_temporal_feed,
    mark_temporal_model,
)
from model_test import INNER_SPLIT_DATE, create_model, sample_training_rows
from semantic_post_features import (
    DEFAULT_TRANSFORMER, SEMANTIC_COLUMNS, build_semantic_post_features,
    attach_semantic_features,
)
from semantic_user_profile import (
    PROFILE_COLUMN, add_user_post_similarity, build_profile_post_vectors,
)
from evaluate_observed_hitrate import audit_unique_posts, probabilities_in_batches

FEATURE_SETS = {
    "user_profile": [*MODEL_FEATURES, PROFILE_COLUMN],
    "user_profile_dist16": [*MODEL_FEATURES, *SEMANTIC_COLUMNS, PROFILE_COLUMN],
}


def quality(y, values):
    """Вычисляет ROC-AUC и Average Precision."""
    return {
        "roc_auc": float(roc_auc_score(y, values)),
        "ap": float(average_precision_score(y, values)),
    }


def choose_epochs(train, validation, feature_names, max_epochs: int, patience: int):
    """Выбирает число эпох по AP на временной Validation."""
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs и patience должны быть положительными")
    fit = sample_training_rows(train)
    X_fit = fit.loc[:, feature_names]
    y_fit = fit.target.astype(np.int8)
    X_val = validation.loc[:, feature_names]
    y_val = validation.target.astype(np.int8)
    if y_fit.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError("В Train/Validation необходимы оба класса")
    model = create_model(X_fit)
    transform = model.named_steps["preprocessor"]
    mlp = model.named_steps["classifier"]
    fit_np = np.asarray(transform.fit_transform(X_fit, y_fit), dtype=np.float32)
    val_np = np.asarray(transform.transform(X_val), dtype=np.float32)
    best_ap, best_epoch, without_gain = -1.0, 1, 0
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            mlp.partial_fit(fit_np, y_fit, classes=np.array([0, 1]))
        else:
            mlp.partial_fit(fit_np, y_fit)
        ap = float(average_precision_score(y_val, mlp.predict_proba(val_np)[:, 1]))
        print(f"  Эпоха {epoch:02d}: Validation AP={ap:.6f}", flush=True)
        if ap > best_ap + 1e-4:
            best_ap, best_epoch, without_gain = ap, epoch, 0
        else:
            without_gain += 1
        if without_gain >= patience:
            break
    del model, transform, mlp, fit_np, val_np, fit
    gc.collect()
    return best_epoch, best_ap


def fit_final(train, feature_names, epochs: int):
    """Обучает вариант MLP на полном Train."""
    if epochs < 1:
        raise ValueError("epochs должен быть положительным")
    fit = sample_training_rows(train)
    X = fit.loc[:, feature_names]
    y = fit.target.astype(np.int8)
    model = create_model(X)
    transform = model.named_steps["preprocessor"]
    mlp = model.named_steps["classifier"]
    fit_np = np.asarray(transform.fit_transform(X, y), dtype=np.float32)
    for epoch in range(epochs):
        if epoch == 0:
            mlp.partial_fit(fit_np, y, classes=np.array([0, 1]))
        else:
            mlp.partial_fit(fit_np, y)
    del fit_np, fit
    gc.collect()
    return model


def load_pickle(path):
    """Загружает локальный pkl для сравнения моделей."""
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        return pickle.load(handle)  # Trusted local model files ONLY.


def main():
    """Обучает экспериментальные модели, не меняя API и старые PKL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-mod", type=int, default=20)
    parser.add_argument("--semantic-cache", type=Path, default=Path("cache/post_semantic_distances.npz"))
    parser.add_argument("--profile-cache", type=Path, default=Path("cache/post_profile_vectors.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("models_personalized_5pct"))
    parser.add_argument("--transformer", default=DEFAULT_TRANSFORMER)
    parser.add_argument("--device", default="cpu", help="cpu или mps для совместимого Mac")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=20000)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--variants", nargs="+", choices=list(FEATURE_SETS), default=list(FEATURE_SETS))
    parser.add_argument("--mlp-baseline", default="models_5pct/model_test.pkl")
    parser.add_argument("--hgb-baseline", default="models_5pct/model_control.pkl")
    parser.add_argument("--semantic-baseline", default="")
    args = parser.parse_args()

    if min(args.user_mod, args.batch_size, args.score_batch_size, args.max_epochs, args.patience) < 1:
        parser.error("Числовые параметры должны быть положительными")
    out_files = {name: args.output_dir / f"model_test_{name}.pkl" for name in args.variants}
    out_csv = args.output_dir / "results_personalization.csv"
    conflicts = [p for p in [*out_files.values(), out_csv] if p.exists()]
    if conflicts:
        parser.error(f"Перезапись файлов запрещена: {conflicts}")

    print("Загружаем историю и тексты постов...", flush=True)
    users, posts, feed = load_data(user_mod=args.user_mod)
    normalized_feed = prepare_temporal_feed(feed)
    vector_posts = build_profile_post_vectors(
        posts, args.profile_cache, args.transformer, args.device, args.batch_size
    )
    sem_posts = build_semantic_post_features(
        posts, args.semantic_cache, args.transformer, args.device, args.batch_size
    ) if "user_profile_dist16" in args.variants or args.semantic_baseline else None

    print("Формируем базовые исторические признаки Train/Test...", flush=True)
    static_posts = build_post_features(posts, fit_pca=False)
    train, test, snapshot = build_temporal_datasets(users, static_posts, normalized_feed)
    print("Вычисляем персональные признаки Train из лайков строго до времени просмотра...", flush=True)
    train = add_user_post_similarity(train, normalized_feed, vector_posts, mode="past")
    print("Вычисляем персональные признаки Test из лайков на 12 декабря...", flush=True)
    test = add_user_post_similarity(
        test, normalized_feed, vector_posts, mode="fixed", cutoff=SPLIT_DATE
    )

    print("Формируем внутреннюю временную Validation...", flush=True)
    inner_snapshot = build_history_snapshot(
        users, static_posts, normalized_feed, split_date=INNER_SPLIT_DATE
    )
    val_events = normalized_feed.loc[
        (normalized_feed.timestamp > INNER_SPLIT_DATE)
        & (normalized_feed.timestamp <= SPLIT_DATE)
    ]
    valid = build_dataset_from_features(
        val_events, inner_snapshot["user_features"], inner_snapshot["post_features"],
        inner_snapshot["user_topic_features"], inner_snapshot["user_cluster_features"],
    )
    print("Вычисляем Validation-профили из лайков на 28 ноября...", flush=True)
    valid = add_user_post_similarity(
        valid, normalized_feed, vector_posts, mode="fixed", cutoff=INNER_SPLIT_DATE
    )
    del inner_snapshot

    if sem_posts is not None:
        print("Добавляем 16 семантических расстояний (из предыдущего кэша)...", flush=True)
        train = attach_semantic_features(train, sem_posts)
        valid = attach_semantic_features(valid, sem_posts)
        test = attach_semantic_features(test, sem_posts)
    if not np.isfinite(train[PROFILE_COLUMN]).all():
        raise RuntimeError("Некорректные значения персонального признака")
    inner_train = train.loc[train.timestamp <= INNER_SPLIT_DATE]
    print(f"Train={len(train):,}, inner Train={len(inner_train):,}, Validation={len(valid):,}, Test={len(test):,}")
    print(f"Доля Train с ненулевым сходством: {(train[PROFILE_COLUMN] != 0).mean():.3%}")
    print(f"Доля Test с ненулевым сходством: {(test[PROFILE_COLUMN] != 0).mean():.3%}")

    validation_results = {}
    best_epochs = {}
    for name in args.variants:
        print(f"\n===== Validation: {name} ({len(FEATURE_SETS[name])} признаков) =====", flush=True)
        best_epoch, val_ap = choose_epochs(
            inner_train, valid, FEATURE_SETS[name], args.max_epochs, args.patience
        )
        validation_results[name] = val_ap
        best_epochs[name] = best_epoch
        print(f"{name}: best_epoch={best_epoch}, Validation AP={val_ap:.6f}", flush=True)
    del valid, inner_train
    gc.collect()

    scores = {}
    rows = []
    for name in args.variants:
        print(f"\n===== Финальная модель: {name} =====", flush=True)
        model = fit_final(train, FEATURE_SETS[name], best_epochs[name])
        mark_temporal_model(model, snapshot)
        model.feature_definition_ += "+" + name + "_v1"
        model.profile_encoder_ = args.transformer
        model.profile_column_ = PROFILE_COLUMN
        model.profile_vector_cache_ = str(args.profile_cache)
        model.profile_history_cutoff_ = str(SPLIT_DATE)
        if "dist16" in name:
            model.semantic_feature_columns_ = list(SEMANTIC_COLUMNS)
        probs = probabilities_in_batches(
            model, test, FEATURE_SETS[name], args.score_batch_size
        )
        scores[name] = probs
        q = quality(test.target, probs)
        rows.append({"model": name, "n_features": len(FEATURE_SETS[name]),
                     "best_epoch": best_epochs[name], "val_ap": validation_results[name], **q})
        print(f"Test {name}: ROC-AUC={q['roc_auc']:.6f}, AP={q['ap']:.6f}", flush=True)
        path = out_files[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Модель сохранена: {path}", flush=True)
        del model
        gc.collect()

    for name, path in (
        ("MLP_baseline", args.mlp_baseline),
        ("MLP_semantic", args.semantic_baseline),
        ("HGB", args.hgb_baseline),
    ):
        if not path:
            continue
        model = load_pickle(path)
        cols = list(model.feature_names_in_)
        if not set(cols).issubset(test.columns):
            raise RuntimeError(f"Для {name} отсутствуют признаки: {set(cols)-set(test.columns)}")
        scores[name] = probabilities_in_batches(model, test, cols, args.score_batch_size)
        q = quality(test.target, scores[name])
        rows.append({"model": name, "n_features": len(cols), "best_epoch": None,
                     "val_ap": None, **q})
        print(f"Test {name}: ROC-AUC={q['roc_auc']:.6f}, AP={q['ap']:.6f}", flush=True)
        del model
    scores["popularity"] = test.post_likes.to_numpy(dtype=np.float64)
    rows.append({"model": "popularity", "n_features": None, "best_epoch": None,
                 "val_ap": None, "roc_auc": None, "ap": None})
    hitrate_summary, detail = audit_unique_posts(test, scores, k=5)
    print("\nObserved HitRate@5: уникальные посты, target первого просмотра", flush=True)
    for row in rows:
        name = row["model"]
        row["observed_hitrate_at_5"] = hitrate_summary["first_hitrate"][name]
        print(f"  {name:20s}: {row['observed_hitrate_at_5']:.6f}", flush=True)
    print("\nЭто диагностический offline HitRate по просмотренным постам, НЕ LMS и НЕ онлайн A/B.")
    results = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    print(f"Сводка сохранена: {out_csv}", flush=True)
    print("Старые модели/таблицы/API не менялись. Новые PKL нельзя ставить в API без новых признаков.")


if __name__ == "__main__":
    main()
