from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


class LayerConfidenceTracker:
    def __init__(self, save_dir: str | Path,
                 csv_name: str = "layer_confidence.csv",
                 figure_name: str = "layer_confidence_heatmap.png") -> None:
        self._save_dir = Path(save_dir)
        self._csv_path = self._save_dir / csv_name
        self._figure_path = self._save_dir / figure_name
        self._epochs: list[int] = []
        self._records: list[np.ndarray] = []
        self._num_layers: Optional[int] = None

    def record(self, epoch_idx: int, snapshot: Optional[torch.Tensor | np.ndarray | Iterable[float]]) -> None:
        if snapshot is None:
            return
        if isinstance(snapshot, torch.Tensor):
            values = snapshot.detach().cpu().float().numpy()
        else:
            values = np.asarray(list(snapshot) if not isinstance(snapshot, np.ndarray) else snapshot, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(f"Layer confidence snapshot must be 1-D, got shape {values.shape}")
        if self._num_layers is None:
            self._num_layers = values.shape[0]
        elif values.shape[0] != self._num_layers:
            raise ValueError("Inconsistent layer count detected while recording layer confidence")
        self._epochs.append(int(epoch_idx))
        self._records.append(values)

    def save(self) -> None:
        if not self._records:
            return
        matrix = np.stack(self._records, axis=0)
        epochs = np.asarray(self._epochs, dtype=np.int32).reshape(-1, 1)
        header = ["epoch"] + [f"layer_{idx}" for idx in range(matrix.shape[1])]
        data = np.concatenate([epochs, matrix], axis=1)
        np.savetxt(self._csv_path, data, delimiter=",", header=",".join(header), comments="")
        self._plot_heatmap(matrix)

    def _plot_heatmap(self, matrix: np.ndarray) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1] * 0.4), max(4, matrix.shape[0] * 0.4)))
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Epoch")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(len(self._epochs)))
        ax.set_yticklabels([str(epoch) for epoch in self._epochs])
        fig.colorbar(im, ax=ax, label="Confidence")
        fig.tight_layout()
        fig.savefig(self._figure_path, dpi=200)
        plt.close(fig)


def plot_layer_confidence_from_csv(csv_path: str | Path, output_path: Optional[str | Path] = None) -> Path:
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.with_name(csv_path.stem + "_heatmap.png")
    output_path = Path(output_path)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("CSV must contain epoch and at least one layer column")
    matrix = data[:, 1:]
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1] * 0.4), max(4, matrix.shape[0] * 0.4)))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Epoch")
    epochs = data[:, 0].astype(int)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(len(epochs)))
    ax.set_yticklabels([str(epoch) for epoch in epochs])
    fig.colorbar(im, ax=ax, label="Confidence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
