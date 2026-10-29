"""Обучение и оценка MLP с временной валидацией."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OneHotEncoder
)

from city_encoder import CityFrequencyEncoder
from connection import load_data
from features import (
    build_history_snapshot,
    build_dataset_from_features,
    build_post_features,
    build_temporal_datasets,
    temporal_split_masks,
    prepare_temporal_feed,
    print_classification_metrics,
    mark_temporal_model,
    SPLIT_DATE,
    MODEL_FEATURES
)


RANDOM_STATE = 42
NEGATIVE_RATIO = 5
INNER_SPLIT_DATE = pd.Timestamp("2021-11-28")
MODEL_PATH = Path("model_test.pkl")


def sample_training_rows(
    dataframe: pd.DataFrame,
    negative_ratio: int = NEGATIVE_RATIO
) -> pd.DataFrame:
    """Оставляет все положительные и не более negative_ratio отрицательных строк."""
    if not isinstance(negative_ratio, int) or isinstance(negative_ratio, bool) or negative_ratio < 1:
        raise ValueError("negative_ratio должен быть положительным целым числом")
    if "target" not in dataframe.columns:
        raise ValueError("В выборке отсутствует target")
    if dataframe["target"].isna().any() or not dataframe["target"].isin([0, 1]).all():
        raise ValueError("target должен содержать только 0 и 1")

    positive = dataframe[
        dataframe["target"] == 1
    ]

    negative = dataframe[
        dataframe["target"] == 0
    ]

    negative_count = min(
        len(negative),
        len(positive) * negative_ratio
    )

    sampled_negative = negative.sample(
        n=negative_count,
        random_state=RANDOM_STATE
    )

    return (
        pd.concat(
            [positive, sampled_negative],
            ignore_index=True
        )
        .sample(
            frac=1,
            random_state=RANDOM_STATE
        )
        .reset_index(drop=True)
    )


def create_model(
    X: pd.DataFrame
) -> Pipeline:
    """Создаёт Pipeline с кодированием признаков и MLP."""
    categorical_columns = (
        X.select_dtypes(
            include=["object", "category", "string"]
        )
        .columns
        .tolist()
    )

    numeric_columns = (
        X.select_dtypes(
            include=["number", "bool"]
        )
        .columns
        .tolist()
    )

    # city кодируем частотой, остальные категории — One-Hot.
    categorical_columns = [col for col in categorical_columns if col != "city"]

    log_columns = [
        "user_actions_count",
        "user_post_count",
        "user_likes_count",
        "user_views_count",
        "text_length",
        "word_count",
        "post_views",
        "post_likes",
        "topic_views",
        "topic_likes",
        "user_topic_views",
        "user_topic_likes"
    ]

    log_columns = [
        column
        for column in log_columns
        if column in numeric_columns
    ]

    regular_numeric_columns = [
        column
        for column in numeric_columns
        if column not in log_columns
    ]

    log_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "log",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    transformers = []

    if log_columns:
        transformers.append(
            ("log_num", log_transformer, log_columns)
        )

    if regular_numeric_columns:
        transformers.append(
            (
                "num",
                numeric_transformer,
                regular_numeric_columns
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_transformer,
                categorical_columns
            )
        )

    # Частота города — один числовой признак вместо множества One-Hot столбцов.
    transformers.append((
        "city",
        Pipeline(steps=[
            ("frequency", CityFrequencyEncoder()),
            ("scaler", StandardScaler()),
        ]),
        ["city"]
    ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False
    )

    classifier = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=5e-4,
        alpha=1e-4,
        batch_size=1024,
        max_iter=30,
        early_stopping=False,
        n_iter_no_change=5,
        tol=1e-4,
        random_state=RANDOM_STATE,
        verbose=True
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )


def train_mlp_with_temporal_stopping(
    train_df: pd.DataFrame,
    users_df: pd.DataFrame,
    base_post_features: pd.DataFrame,
    feed_df: pd.DataFrame,
    max_epochs: int = 30,
    patience: int = 3
):
    """Выбирает число эпох по AP на временной Validation и обучает финальную MLP."""
    if not isinstance(max_epochs, int) or isinstance(max_epochs, bool) or max_epochs < 1:
        raise ValueError("max_epochs должен быть положительным целым числом")
    if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
        raise ValueError("patience должен быть положительным целым числом")
    if not INNER_SPLIT_DATE < SPLIT_DATE:
        raise ValueError("Внутренняя граница должна предшествовать SPLIT_DATE")

    inner_train = train_df.loc[
        train_df["timestamp"] <= INNER_SPLIT_DATE
    ]
    fit_rows = sample_training_rows(inner_train)

    # Признаки Validation фиксируются на 28 ноября, до начала проверочного периода.
    inner_snapshot = build_history_snapshot(
        users_df, base_post_features, feed_df,
        split_date=INNER_SPLIT_DATE
    )
    val_events = feed_df.loc[
        (feed_df["timestamp"] > INNER_SPLIT_DATE)
        & (feed_df["timestamp"] <= SPLIT_DATE)
    ]
    val_df = build_dataset_from_features(
        val_events,
        inner_snapshot["user_features"],
        inner_snapshot["post_features"],
        inner_snapshot["user_topic_features"],
        inner_snapshot["user_cluster_features"]
    )

    if fit_rows.empty or val_df.empty:
        raise ValueError("Пустая внутренняя обучающая или проверочная выборка")

    X_fit = fit_rows[MODEL_FEATURES]
    y_fit = fit_rows["target"].astype(np.int8)
    X_val = val_df[MODEL_FEATURES]
    y_val = val_df["target"].astype(np.int8)

    if y_fit.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError("Во внутреннем Train и Validation должны быть оба класса")

    selection_model = create_model(X_fit)
    preprocessor = selection_model.named_steps["preprocessor"]
    classifier = selection_model.named_steps["classifier"]

    X_fit_processed = np.asarray(
        preprocessor.fit_transform(X_fit, y_fit), dtype=np.float32
    )
    X_val_processed = np.asarray(
        preprocessor.transform(X_val), dtype=np.float32
    )

    best_ap = -1.0
    best_epoch = 1
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            classifier.partial_fit(
                X_fit_processed, y_fit, classes=np.array([0, 1])
            )
        else:
            classifier.partial_fit(X_fit_processed, y_fit)

        probabilities = classifier.predict_proba(X_val_processed)[:, 1]
        ap = average_precision_score(y_val, probabilities)
        print(f"Epoch {epoch:02d}: temporal Validation AP = {ap:.6f}")

        if ap > best_ap + 1e-4:
            best_ap = ap
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping: лучшая эпоха {best_epoch}")
            break

    print(f"Лучшая эпоха: {best_epoch}, Validation AP: {best_ap:.6f}")

    # Обучаем новую модель на полном Train; Test не используется.
    final_rows = sample_training_rows(train_df)
    X_final = final_rows[MODEL_FEATURES]
    y_final = final_rows["target"].astype(np.int8)
    if y_final.nunique() != 2:
        raise ValueError("В полном Train должны быть оба класса")

    final_model = create_model(X_final)
    final_preprocessor = final_model.named_steps["preprocessor"]
    final_classifier = final_model.named_steps["classifier"]
    X_final_processed = np.asarray(
        final_preprocessor.fit_transform(X_final, y_final), dtype=np.float32
    )

    for epoch in range(best_epoch):
        if epoch == 0:
            final_classifier.partial_fit(
                X_final_processed, y_final, classes=np.array([0, 1])
            )
        else:
            final_classifier.partial_fit(X_final_processed, y_final)

    print(f"Финальная MLP обучена за {best_epoch} эпох(и)")
    return final_model, best_epoch


def main():
    """Обучает MLP, оценивает качество на Test и сохраняет модель."""

    parser = argparse.ArgumentParser(
        description="Обучение MLP без утечки: локальный snapshot и временный Test"
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--output-model", default=str(MODEL_PATH))
    parser.add_argument("--overwrite-model", action="store_true")
    args = parser.parse_args()

    if not args.train:
        parser.error("Для обучения требуется --train")

    output_model = Path(args.output_model)
    if output_model.exists() and not args.overwrite_model:
        parser.error("Модель уже существует; перезапись требует --overwrite-model")

    users_df, posts_df, feed_df = load_data()

    temporal_split_masks(feed_df)

    base_post_features = build_post_features(
        posts_df=posts_df,
        fit_pca=False
    )

    # Train использует прошлые события; Test — признаки на дату SPLIT_DATE.
    train_df, test_df, snapshot = build_temporal_datasets(
        users_df, base_post_features, feed_df
    )

    normalized_feed = prepare_temporal_feed(feed_df)
    print("Split time:", SPLIT_DATE)
    print(f"Train: rows={len(train_df)}, positive_share={train_df['target'].mean():.6f}")
    print(f"Test: rows={len(test_df)}, positive_share={test_df['target'].mean():.6f}")


    # Число эпох выбирается без независимого Test.
    evaluation_model, best_epoch = train_mlp_with_temporal_stopping(
        train_df, users_df, base_post_features, normalized_feed
    )

    # ROC-AUC и AP по полному Train и независимому Test.
    print_classification_metrics(evaluation_model, train_df, test_df)


    # Используем модель, переобученную на всём Train с выбранным числом эпох.
    final_model = evaluation_model

    mark_temporal_model(final_model, snapshot)

    output_model.parent.mkdir(parents=True, exist_ok=True)
    with output_model.open("wb" if args.overwrite_model else "xb") as file:
        pickle.dump(
            final_model,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(f"Модель сохранена: {output_model}")
    print("Для API нужны таблицы признаков из того же snapshot, что и модель.")


if __name__ == "__main__":
    main()
