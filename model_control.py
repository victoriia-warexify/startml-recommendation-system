"""Обучение и оценка контрольной модели HistGradientBoostingClassifier."""

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from city_encoder import CityFrequencyEncoder
from connection import load_data
from features import (
    MODEL_FEATURES,
    SPLIT_DATE,
    build_post_features,
    build_temporal_datasets,
    mark_temporal_model,
    print_classification_metrics,
    temporal_split_masks,
)


def create_model(X):
    """Собирает предобработку признаков и модель HGB."""
    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    # Город кодируется частотой в Train, другие категории — One-Hot.
    categorical_columns = [col for col in categorical_columns if col != "city"]

    log_columns = [
        "user_actions_count",
        "user_post_count",
        "user_likes_count",
        "user_views_count",
        "text_length",
        "word_count",
        "unique_word_count",
    ]
    log_columns = [col for col in log_columns if col in numeric_columns]
    regular_numeric_columns = [
        col for col in numeric_columns if col not in log_columns
    ]

    log_numeric_transformer = Pipeline(
        steps=[
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
        ]
    )
    regular_numeric_transformer = "passthrough"
    categorical_transformer = Pipeline(
        steps=[
            ("onehot_encoder", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("log_num", log_numeric_transformer, log_columns),
            ("num", regular_numeric_transformer, regular_numeric_columns),
            ("one_hot", categorical_transformer, categorical_columns),
            ("city", CityFrequencyEncoder(), ["city"]),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=42,
            )),
        ]
    )


def main():
    """Обучает HGB, оценивает качество и сохраняет модель."""
    parser = argparse.ArgumentParser(
        description="Обучение Control на временном Train/Test"
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--output-model", default="model_control.pkl")
    parser.add_argument("--overwrite-model", action="store_true")
    args = parser.parse_args()

    if not args.train:
        parser.error("Для обучения требуется --train")

    output_model = Path(args.output_model)
    if output_model.exists() and not args.overwrite_model:
        parser.error("Модель уже существует. Для замены нужен --overwrite-model")

    users_df, posts_df, feed_df = load_data()
    temporal_split_masks(feed_df)

    # Текстовые признаки рассчитываются отдельно, без target и истории лайков.
    post_features = build_post_features(posts_df=posts_df, fit_pca=False)

    # Train получает признаки из прошлого каждого события, Test — из snapshot.
    train_df, test_df, snapshot = build_temporal_datasets(
        users_df, post_features, feed_df
    )

    y_train = train_df["target"].astype(int)
    y_test = test_df["target"].astype(int)
    X_train = train_df[MODEL_FEATURES].copy()
    X_test = test_df[MODEL_FEATURES].copy()

    if y_train.nunique() < 2:
        raise ValueError("В обучающем периоде должны быть оба класса")

    print(f"Train <= {SPLIT_DATE}, Test > {SPLIT_DATE}")
    print(f"Train: rows={len(train_df)}, positive_share={y_train.mean():.6f}")
    print(f"Test: rows={len(test_df)}, positive_share={y_test.mean():.6f}")

    model = create_model(X_train)
    model.fit(X_train, y_train)

    print_classification_metrics(model, train_df, test_df)
    print("Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))

    # Метаданные понадобятся API для проверки исторического snapshot.
    mark_temporal_model(model, snapshot)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    with output_model.open("wb" if args.overwrite_model else "xb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Модель сохранена в файл {output_model}")

    # Сравниваем предсказания до и после сохранения, а не только наличие методов.
    with output_model.open("rb") as file:
        loaded_model = pickle.load(file)
    sample = X_test.head()
    np.testing.assert_allclose(
        loaded_model.predict_proba(sample), model.predict_proba(sample)
    )
    print("Проверка загрузки модели: OK")
    print("Для API нужны таблицы признаков из того же snapshot, что и модель.")


if __name__ == "__main__":
    main()
