import numpy as np

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional dependency
    lgb = None


class PerformancePredictor:
    """LightGBM regressor mapping mixture weights -> proxy performance."""

    def __init__(self):
        self.model = None
        self._weights = None
        self._performances = None
        self.params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": 4,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "min_data_in_leaf": 5,
            "verbose": -1,
        }

    def fit(self, mixture_weights: np.ndarray, performances: np.ndarray):
        if lgb is None:
            raise ImportError("lightgbm is required for the performance predictor.")
        self._weights = np.asarray(mixture_weights, dtype=float)
        self._performances = np.asarray(performances, dtype=float)

        train_data = lgb.Dataset(self._weights, label=self._performances)
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30)],
        )

    def predict(self, mixture_weights: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Predictor not fitted.")
        return self.model.predict(np.asarray(mixture_weights, dtype=float))

    def update(self, new_weights: np.ndarray, new_performances: np.ndarray):
        if self._weights is None:
            return self.fit(new_weights, new_performances)
        all_weights = np.vstack([self._weights, new_weights])
        all_perf = np.concatenate([self._performances, new_performances])
        return self.fit(all_weights, all_perf)
