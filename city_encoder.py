"""Частотное кодирование города без использования целевой переменной."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CityFrequencyEncoder(TransformerMixin, BaseEstimator):
    """Заменяет city долей строк этого города в обучающей выборке.

    Для города, которого не было при обучении, возвращает 0.
    Словарь частот сохраняется внутри sklearn Pipeline.
    """

    @staticmethod
    def _cities(X):
        """Извлекает единственный столбец и заменяет пропуски на unknown."""
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("CityFrequencyEncoder ожидает один столбец city")
            values = X.iloc[:, 0]
        else:
            array = np.asarray(X)
            if array.ndim != 2 or array.shape[1] != 1:
                raise ValueError("CityFrequencyEncoder ожидает один столбец city")
            values = pd.Series(array[:, 0])
        return values.fillna("unknown").astype(str)

    def fit(self, X, y=None):
        """Запоминает частоты городов на обучающей выборке, не используя target."""
        self.frequencies_ = self._cities(X).value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        """Возвращает частоты городов; неизвестным присваивает 0."""
        check_is_fitted(self, "frequencies_")
        frequencies = self._cities(X).map(self.frequencies_).fillna(0.0)
        return frequencies.to_numpy(dtype=np.float32).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        """Возвращает имя выходного признака."""
        return np.array(["city_frequency"], dtype=object)
