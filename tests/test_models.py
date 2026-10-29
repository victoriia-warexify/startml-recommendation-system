
"""Тесты обучения HGB и MLP без подключения к PostgreSQL."""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from features import MODEL_FEATURES
from city_encoder import CityFrequencyEncoder
from model_control import create_model as create_control
from model_test import create_model as create_test
from model_test import sample_training_rows


def example_features(n_rows=60):
    """Искусственная таблица со всеми 32 признаками проекта."""
    rows = []

    for i in range(n_rows):
        rows.append({
            "hour": i % 24,
            "day_of_week": i % 7,
            "month": 11,
            "is_weekend": int(i % 7 >= 5),
            "gender": i % 2,
            "age": 20 + i % 30,
            "country": "RU" if i % 2 else "KZ",
            "city": "Moscow" if i % 3 else "Astana",
            "exp_group": i % 3,
            "os": "iOS" if i % 2 else "Android",
            "source": "ads" if i % 3 else "organic",
            "topic": "tech" if i % 2 else "food",
            "user_actions_count": i + 2,
            "user_post_count": i + 1,
            "user_likes_count": i // 4,
            "user_views_count": i + 1,
            "post_views": i + 10,
            "post_likes": i // 4,
            "post_ctr": 0.25,
            "post_popularity": np.log1p(i + 10),
            "topic_views": i + 20,
            "topic_likes": i // 3,
            "topic_ctr": 0.3,
            "user_topic_views": i + 1,
            "user_topic_likes": i // 5,
            "user_topic_ctr": 0.2,
            "text_cluster": "tech:c00" if i % 2 else "food:c00",
            "text_length": 10 + i,
            "word_count": 3 + i // 2,
            "user_cluster_views": i + 1,
            "user_cluster_likes": i // 6,
            "user_cluster_ctr": 0.22,
        })

    X = pd.DataFrame(rows)
    assert list(X.columns) == MODEL_FEATURES

    y = pd.Series(
        (np.arange(n_rows) % 3 == 0).astype(np.int8)
    )

    return X, y


def test_negative_sampling_is_deterministic_and_limits_ratio():
    """Проверяем ограничение отрицательных примеров до 5:1."""
    data = pd.DataFrame({
        "row_id": range(110),
        "target": [1] * 10 + [0] * 100
    })

    sampled = sample_training_rows(data)

    assert len(sampled) == 60
    assert int(sampled.target.sum()) == 10
    assert int((sampled.target == 0).sum()) == 50

    pd.testing.assert_frame_equal(
        sampled,
        sample_training_rows(data)
    )

    # Исходная таблица не меняется.
    assert data.shape == (110, 2)


def test_negative_sampling_uses_all_negatives_when_there_are_few():
    """Если отрицательных примеров мало, сохраняем их все."""
    data = pd.DataFrame({
        "row_id": range(15),
        "target": [1] * 10 + [0] * 5
    })

    sampled = sample_training_rows(data)

    assert len(sampled) == 15
    assert set(sampled.row_id) == set(data.row_id)


@pytest.mark.parametrize(
    "factory",
    [create_control, create_test],
    ids=["HGB", "MLP"]
)
def test_models_keep_features_and_predict_probabilities(factory):
    """Проверяем обе модели, вероятности и сериализацию."""
    X, y = example_features()

    model = factory(X)

    # Сокращаем обучение только для тестов.
    model.set_params(classifier__max_iter=1)

    if factory is create_test:
        model.set_params(classifier__batch_size=32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)

    # Порядок признаков не должен измениться.
    assert list(model.feature_names_in_) == MODEL_FEATURES

    proba = model.predict_proba(X.iloc[:4])

    assert proba.shape == (4, 2)
    assert np.isfinite(proba).all()

    np.testing.assert_allclose(
        proba.sum(axis=1),
        1.0
    )

    assert ((proba >= 0) & (proba <= 1)).all()

    # Новые категории не должны приводить к ошибкам.
    new_users = X.iloc[:2].copy()
    new_users["city"] = "Other city"
    new_users["topic"] = "new_topic"

    assert np.isfinite(
        model.predict_proba(new_users)
    ).all()

    # Проверяем сохранение и повторную загрузку.
    restored = pickle.loads(pickle.dumps(model))

    np.testing.assert_allclose(
        restored.predict_proba(new_users),
        model.predict_proba(new_users)
    )


def test_city_frequency_uses_train_only_and_handles_unknowns():
    """Проверяем частотное кодирование городов."""
    train = pd.DataFrame({
        "city": ["Moscow", "Moscow", "Kazan", None]
    })

    val = pd.DataFrame({
        "city": ["Moscow", "Kazan", "Other", None]
    })

    encoder = CityFrequencyEncoder().fit(train)

    np.testing.assert_allclose(
        encoder.transform(val).ravel(),
        [0.5, 0.25, 0.0, 0.25]
    )

    assert encoder.get_feature_names_out().tolist() == [
        "city_frequency"
    ]

    restored = pickle.loads(pickle.dumps(encoder))

    assert restored.transform(val).shape == (4, 1)


@pytest.mark.parametrize(
    "factory",
    [create_control, create_test],
    ids=["HGB", "MLP"]
)
def test_city_creates_only_one_output_feature(factory):
    """Проверяем, что city не преобразуется в One-Hot."""
    X, _ = example_features()

    preprocessor = factory(X).named_steps["preprocessor"]

    X_encoded = preprocessor.fit_transform(X)
    names = preprocessor.get_feature_names_out().tolist()

    # После Frequency Encoding должен быть один столбец.
    assert sum(
        name.endswith("city_frequency")
        for name in names
    ) == 1

    # One-Hot столбцов для city быть не должно.
    assert not any(
        "one_hot__city_" in name
        or "categorical__city_" in name
        for name in names
    )

    assert X_encoded.shape[1] == len(names)


@pytest.mark.parametrize("ratio", [0, -1, 1.5, "5", True])
def test_negative_sampling_rejects_invalid_ratio(ratio):
    """Некорректное соотношение классов не должно приниматься."""
    rows = pd.DataFrame({"target": [0, 1]})
    with pytest.raises(ValueError, match="negative_ratio"):
        sample_training_rows(rows, negative_ratio=ratio)


@pytest.mark.parametrize("target", [[0, 1, 2], [0, 1, None]])
def test_negative_sampling_rejects_invalid_target(target):
    """В обучающей выборке допустимы только метки 0 и 1."""
    rows = pd.DataFrame({"target": target})
    with pytest.raises(ValueError, match="target"):
        sample_training_rows(rows)


@pytest.mark.parametrize("max_epochs, patience", [(0, 3), (1, 0), (-1, 1), (True, 2)])
def test_temporal_training_rejects_invalid_parameters(max_epochs, patience):
    """Неправильные параметры останавливают обучение до подготовки данных."""
    from model_test import train_mlp_with_temporal_stopping
    with pytest.raises(ValueError, match="max_epochs|patience"):
        train_mlp_with_temporal_stopping(None, None, None, None,
                                         max_epochs=max_epochs, patience=patience)


def test_mlp_temporal_stopping_uses_inner_validation(monkeypatch):
    """MLP выбирает эпохи на временной Validation, не используя Test."""
    from features import build_temporal_datasets, prepare_temporal_feed
    import model_test
    from model_test import train_mlp_with_temporal_stopping

    # В тесте используем небольшой batch_size, настройки основного обучения не меняем.
    original_factory = model_test.create_model

    def create_small_batch_model(X):
        model = original_factory(X)
        model.set_params(classifier__batch_size=2, classifier__verbose=False)
        return model

    monkeypatch.setattr(model_test, "create_model", create_small_batch_model)

    users = pd.DataFrame([{
        "user_id": 1, "gender": 0, "age": 26, "country": "RU",
        "city": "Kazan", "exp_group": 1, "os": "iOS", "source": "ads",
    }])
    posts = pd.DataFrame([
        {"post_id": 10, "topic": "tech", "text_cluster": "tech:c00", "text_length": 12, "word_count": 3},
        {"post_id": 11, "topic": "food", "text_cluster": "food:c00", "text_length": 18, "word_count": 4},
    ])
    events = pd.DataFrame([
        {"user_id": 1, "post_id": post_id, "timestamp": timestamp,
         "action": "view", "target": target}
        for timestamp, post_id, target in [
            ("2021-11-20 10:00", 10, 0),
            ("2021-11-21 10:00", 11, 1),
            ("2021-11-22 10:00", 10, 0),
            ("2021-11-23 10:00", 11, 1),
            ("2021-12-02 10:00", 10, 0),
            ("2021-12-03 10:00", 11, 1),
            ("2021-12-14 10:00", 10, 1),
        ]
    ])
    train, test, _ = build_temporal_datasets(users, posts, events)
    assert len(train) == 6
    assert len(test) == 1

    model, best_epoch = train_mlp_with_temporal_stopping(
        train, users, posts, prepare_temporal_feed(events),
        max_epochs=2, patience=2,
    )
    assert best_epoch in (1, 2)
    assert list(model.feature_names_in_) == MODEL_FEATURES
    predictions = model.predict_proba(test[MODEL_FEATURES])
    assert predictions.shape == (1, 2)
    assert np.isfinite(predictions).all()
