"""Сравнение наборов признаков HGB и MLP на временной Validation.

Train: события до 28 ноября 2021 года включительно.
Validation: с 28 ноября по 12 декабря 2021 года (левая граница исключена).
Признаки Validation используют историю только до 28 ноября.
Независимый Test после 12 декабря не затрагивается.

По умолчанию проверяются два набора (27 и 28 признаков) на 1% пользователей.
Существующие модели и таблицы PostgreSQL не изменяются.
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from connection import load_data
from features import (
    MODEL_FEATURES,
    SPLIT_DATE,
    build_post_features,
    build_temporal_datasets,
    normalize_event_times,
)
from model_control import create_model as create_hgb
from model_test import create_model as create_mlp
from model_test import sample_training_rows

INNER_SPLIT_DATE = pd.Timestamp("2021-11-28")
FEATURE_SETS = {
    "history_26": MODEL_FEATURES[:26],
    "history_cluster_27": MODEL_FEATURES[:26] + ["text_cluster"],
    "history_length_28": MODEL_FEATURES[:26] + ["text_length", "word_count"],
    "history_text_29": MODEL_FEATURES[:29],
    "full_32": MODEL_FEATURES,
}
NEW_FEATURE_SETS = ["history_cluster_27", "history_length_28"]


def load_sample(user_mod: int):
    """Загружает события выбранной когорты и полные справочники пользователей и постов."""
    users, posts, feed = load_data(user_mod=user_mod)
    print(f"Загружено событий: {len(feed):,}; когорта user_id % {user_mod} = 0")
    return users, posts, feed


def prepare_validation(users: pd.DataFrame, posts: pd.DataFrame, feed: pd.DataFrame):
    """Формирует временные Train и Validation без событий независимого Test."""
    if not INNER_SPLIT_DATE < SPLIT_DATE:
        raise ValueError("Внутренняя дата должна быть раньше Test")

    feed = feed.copy()
    feed["timestamp"] = normalize_event_times(feed["timestamp"])
    # Декабрьский независимый Test не должен влиять на подбор признаков.
    development = feed.loc[feed["timestamp"] <= SPLIT_DATE].copy()
    print("Строим текстовые и past-only признаки для временной Validation...")
    base_posts = build_post_features(posts, fit_pca=False)
    train, val, _snapshot = build_temporal_datasets(
        users, base_posts, development, split_date=INNER_SPLIT_DATE
    )
    if train.empty or val.empty:
        raise ValueError("Train/Validation пусты: проверьте даты и объём выборки")
    if train.target.nunique() != 2 or val.target.nunique() != 2:
        raise ValueError("Для ROC-AUC нужны оба класса в Train и Validation")
    print(
        f"Train <= {INNER_SPLIT_DATE}: {len(train):,}, доля target=1 {train.target.mean():.4f}\n"
        f"Validation ({INNER_SPLIT_DATE}, {SPLIT_DATE}]: {len(val):,}, "
        f"доля target=1 {val.target.mean():.4f}"
    )
    return train, val


def scores(y_true, probabilities):
    """Считает ROC-AUC и Average Precision на Validation."""
    return {
        "val_roc_auc": float(roc_auc_score(y_true, probabilities)),
        "val_ap": float(average_precision_score(y_true, probabilities)),
    }


def evaluate_hgb(train, val, cols):
    """Обучает HGB на выбранных признаках и оценивает на Validation."""
    model = create_hgb(train[cols])
    model.fit(train[cols], train.target.astype(int))
    result = scores(val.target, model.predict_proba(val[cols])[:, 1])
    del model
    return result


def evaluate_mlp(train, val, cols, max_epochs: int, patience: int):
    """Выбирает лучшую эпоху MLP по AP на полной временной Validation.

    Для обучения используются все положительные примеры и не более пяти
    отрицательных на один положительный.
    """
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs и patience должны быть положительными")
    sampled = sample_training_rows(train)
    X_fit = sampled.loc[:, cols]
    y_fit = sampled.target.astype(np.int8)
    X_val = val.loc[:, cols]
    y_val = val.target.astype(np.int8)

    pipeline = create_mlp(X_fit)
    processor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_fit_encoded = np.asarray(processor.fit_transform(X_fit, y_fit), dtype=np.float32)
    X_val_encoded = np.asarray(processor.transform(X_val), dtype=np.float32)

    best_ap = -1.0
    best_epoch = 1
    best_metrics = None
    bad_epochs = 0
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            classifier.partial_fit(X_fit_encoded, y_fit, classes=np.array([0, 1]))
        else:
            classifier.partial_fit(X_fit_encoded, y_fit)
        proba = classifier.predict_proba(X_val_encoded)[:, 1]
        metrics = scores(y_val, proba)
        print(f"    Epoch {epoch:02d}: AP={metrics['val_ap']:.6f}, "
              f"ROC-AUC={metrics['val_roc_auc']:.6f}")
        if metrics["val_ap"] > best_ap + 1e-4:
            best_ap = metrics["val_ap"]
            best_metrics = metrics
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    result = {**best_metrics, "best_epoch": best_epoch}
    del pipeline, classifier, processor, X_fit_encoded, X_val_encoded
    return result


def main():
    """Запускает сравнение моделей и сохраняет результаты в CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["hgb", "mlp"], default=["hgb", "mlp"])
    parser.add_argument("--sets", nargs="+", choices=list(FEATURE_SETS),
                        default=NEW_FEATURE_SETS,
                        help="Наборы для обучения (по умолчанию только новые: 27 и 28)")
    parser.add_argument("--user-mod", type=int, default=100,
                        help="100 = ~1%% пользователей, 50 = ~2%%, 20 = ~5%%")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("ablation_results.csv"))
    args = parser.parse_args()
    if args.max_epochs < 1 or args.patience < 1:
        parser.error("--max-epochs и --patience должны быть положительными")
    if args.user_mod < 1:
        parser.error("--user-mod должен быть >= 1")
    if args.output.exists():
        parser.error(f"{args.output} уже существует; задайте другой --output")

    for name, columns in FEATURE_SETS.items():
        if len(columns) != int(name.rsplit("_", 1)[1]) or len(set(columns)) != len(columns):
            parser.error(f"Некорректный состав набора {name}: {columns}")
        if not set(columns).issubset(MODEL_FEATURES):
            parser.error(f"Набор {name} содержит неизвестные признаки")

    users, posts, feed = load_sample(args.user_mod)
    train, val = prepare_validation(users, posts, feed)
    del users, posts, feed
    gc.collect()

    rows = []
    for family in args.models:
        for name in args.sets:
            cols = FEATURE_SETS[name]
            print(f"\n===== {family.upper()} / {name} / {len(cols)} признаков =====", flush=True)
            start = time.perf_counter()
            if family == "hgb":
                metrics = evaluate_hgb(train, val, cols)
            else:
                metrics = evaluate_mlp(train, val, cols, args.max_epochs, args.patience)
            row = {
                "model": family.upper(),
                "feature_set": name,
                "n_features": len(cols),
                "train_rows": len(train),
                "val_rows": len(val),
                **metrics,
                "seconds": round(time.perf_counter() - start, 2),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(f"Результат: AP={row['val_ap']:.6f}, "
                  f"ROC-AUC={row['val_roc_auc']:.6f}; "
                  f"{row['seconds']:.1f} с; сохранено {args.output}", flush=True)
            gc.collect()

    results = pd.DataFrame(rows)
    print("\n========== СРАВНЕНИЕ НА ОДНОЙ VALIDATION ==========", flush=True)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nПобедителей выбираем по Validation AP. Независимый Test пока не трогаем.")
    print("Ни один существующий PKL не изменён.")


if __name__ == "__main__":
    main()
