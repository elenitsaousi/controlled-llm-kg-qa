from abc import ABC, abstractmethod
import os
from typing import Dict, Iterable, List

import numpy as np
import joblib

from ranking.feature_config import FEATURE_NAMES


DEFAULT_XGB_MODEL = os.path.join("ranking", "models", "xgb_ranker.json")
DEFAULT_LOGISTIC_MODEL = os.path.join("ranking", "models", "logistic_ranker.joblib")


class QueryRanker(ABC):
    def __init__(self, feature_names: Iterable[str] = FEATURE_NAMES):
        self.feature_names = list(feature_names)

    def _to_matrix(self, feature_dicts: List[Dict[str, float]]) -> np.ndarray:
        if not feature_dicts:
            return np.empty((0, len(self.feature_names)), dtype=float)
        rows = []
        for feats in feature_dicts:
            rows.append([float(feats[name]) for name in self.feature_names])
        return np.array(rows, dtype=float)

    @abstractmethod
    def score(self, feature_dicts: List[Dict[str, float]]) -> np.ndarray:
        raise NotImplementedError


class LogisticRanker(QueryRanker):
    def __init__(self, model_path: str = DEFAULT_LOGISTIC_MODEL):
        super().__init__()
        self.model_path = model_path
        self.model = joblib.load(model_path)

    def score(self, feature_dicts: List[Dict[str, float]]) -> np.ndarray:
        X = self._to_matrix(feature_dicts)
        if X.size == 0:
            return np.array([])
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.decision_function(X)


class XGBRanker(QueryRanker):
    def __init__(self, model_path: str = DEFAULT_XGB_MODEL):
        super().__init__()
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost is not installed") from exc
        self._xgb = xgb
        self.model_path = model_path
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def score(self, feature_dicts: List[Dict[str, float]]) -> np.ndarray:
        X = self._to_matrix(feature_dicts)
        if X.size == 0:
            return np.array([])
        dmat = self._xgb.DMatrix(X)
        return self.model.predict(dmat)
