"""AdaptiveDEER-MLP / GRU and a sklearn-free StoppingController wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import FEATURE_COLS, engineer_trajectory_features, features_from_deer_checkpoints


class _StandardScaler:
    """Minimal StandardScaler (avoids sklearn/scipy DLL issues on some hosts)."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray) -> "_StandardScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.18):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class AdaptiveDEER_MLP(nn.Module):
    """Residual MLP stopping controller (~14.5K params in notebook config)."""

    def __init__(self, in_features: int = 12, hidden_dim: int = 64, dropout: float = 0.18):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.res1 = ResidualBlock(hidden_dim, dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.res1(h)
        h = self.res2(h)
        return self.head(h).squeeze(-1)


class AdaptiveDEER_GRU(nn.Module):
    def __init__(self, in_features: int = 12, hidden_dim: int = 48):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_dim, num_layers=2, batch_first=True, dropout=0.15)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 24), nn.SiLU(), nn.Linear(24, 1))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x_seq)
        return self.head(out[:, -1, :]).squeeze(-1)


class AsymmetricCostStoppingLoss(nn.Module):
    def __init__(self, fp_penalty: float = 2.35, gamma: float = 1.5):
        super().__init__()
        self.fp_penalty = fp_penalty
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        focal_weight = (1.0 - pt) ** self.gamma
        asym_weight = torch.where(targets == 0, self.fp_penalty, 1.0)
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        return torch.mean(asym_weight * focal_weight * bce)


class StoppingController:
    """Train / save / load AdaptiveDEER-MLP and expose should_stop(cps, rec)."""

    def __init__(self, threshold: float = 0.62, device: Optional[str] = None):
        self.threshold = threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AdaptiveDEER_MLP(in_features=len(FEATURE_COLS)).to(self.device)
        self.scaler = _StandardScaler()
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 2.5e-3,
        fp_penalty: float = 2.35,
    ) -> List[float]:
        self.scaler.fit(X)
        Xs = self.scaler.transform(X).astype(np.float32)
        y = np.asarray(y, dtype=np.float32)
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(Xs), torch.from_numpy(y)
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        crit = AsymmetricCostStoppingLoss(fp_penalty=fp_penalty)
        losses: List[float] = []
        self.model.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            n = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item()) * len(xb)
                n += len(xb)
            losses.append(epoch_loss / max(n, 1))
        self._fitted = True
        return losses

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Controller not fitted / loaded")
        Xs = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(Xs).to(self.device))
            return torch.sigmoid(logits).cpu().numpy()

    def should_stop(self, checkpoints: Sequence[dict], rec=None) -> bool:
        """Vishnu simulate_policy-compatible callback."""
        if not checkpoints:
            return False
        pid = str((rec or {}).get("idx", (rec or {}).get("problem_id", "q")))
        raw = features_from_deer_checkpoints(list(checkpoints), problem_id=pid)
        feat, cols = engineer_trajectory_features(raw)
        x = feat[cols].values.astype(np.float32)
        p = float(self.predict_proba(x)[-1])
        return p >= self.threshold

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "scaler_mean": self.scaler.mean_,
                "scaler_scale": self.scaler.scale_,
                "threshold": self.threshold,
                "feature_cols": FEATURE_COLS,
            },
            path,
        )

    def load(self, path: str | Path) -> "StoppingController":
        try:
            blob = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location=self.device)
        self.model.load_state_dict(blob["state_dict"])
        self.scaler.mean_ = blob["scaler_mean"]
        self.scaler.scale_ = blob["scaler_scale"]
        self.scaler.n_features_in_ = len(blob["scaler_mean"])
        self.threshold = float(blob.get("threshold", self.threshold))
        self._fitted = True
        return self
