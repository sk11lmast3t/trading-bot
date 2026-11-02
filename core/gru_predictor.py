"""
GRU Deep Learning Predictor Module
Extracted from crypto_strategy-main and modernized as a reusable neural forecaster.
"""
import os
import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import GRU, Dropout, Dense
    from tensorflow.keras.regularizers import l2
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logger.warning("TensorFlow or scikit-learn not installed. GRU predictor runs in fallback mode.")


class GRUPredictor:
    def __init__(self, model_path: str = "models/crypto_gru_model.h5",
                 scaler_path: str = "models/crypto_gru_model_scaler.pkl",
                 lookback: int = 20):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.lookback = lookback
        self.model = None
        self.scaler = None
        self._load()

    def _load(self):
        if not HAS_TF:
            return
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded GRU model from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading GRU model: {e}")

    def predict_next_price(self, df: pd.DataFrame) -> Optional[float]:
        if not HAS_TF or self.model is None or self.scaler is None:
            return None
        if len(df) < self.lookback:
            return None

        try:
            # Select 5 standard OHLCV features
            feat_cols = ['open', 'high', 'low', 'close', 'volume']
            avail_cols = [c for c in feat_cols if c in df.columns]
            if len(avail_cols) < 5:
                return None

            sub_df = df[avail_cols].tail(self.lookback).astype(float)
            local_scaler = MinMaxScaler()
            scaled = local_scaler.fit_transform(sub_df)
            X = scaled.reshape(1, self.lookback, sub_df.shape[1])
            pred_scaled = self.model.predict(X, verbose=0)

            # Invert scaling on close price (column index 3: 'close')
            close_idx = avail_cols.index('close') if 'close' in avail_cols else 3
            dummy = np.zeros((1, sub_df.shape[1]))
            dummy[0, close_idx] = pred_scaled[0, 0]
            pred_price = local_scaler.inverse_transform(dummy)[0, close_idx]
            return float(pred_price)
        except Exception as e:
            logger.debug(f"GRU prediction error: {e}")
            return None
