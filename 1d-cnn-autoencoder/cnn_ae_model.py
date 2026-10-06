"""1D CNN Autoencoder for the verified 16 x 3 KAMP press windows.

Run ``python cnn_ae_model.py --data cnn_ae_data.npz --out cnn_ae_results``.
Use ``--check-data`` to inspect the NPZ without installing TensorFlow.
Requires numpy and (for training) tensorflow, scikit-learn, matplotlib.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

KEYS = ("X_train", "X_val", "X_cal", "X_test_normal", "X_test_anomaly")
EXPECTED = dict(zip(KEYS, (610, 160, 116, 157, 28)))
CHANNELS = ("AI0_Vibration", "AI1_Vibration", "AI2_Current")


def load_data(path: Path, strict: bool = True) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"Missing NPZ arrays: {sorted(missing)}")
        arrays = {k: archive[k] for k in KEYS}
    for name, arr in arrays.items():
        if arr.ndim != 3 or arr.shape[1:] != (16, 3) or not np.isfinite(arr).all():
            raise ValueError(f"{name}: expected finite (N, 16, 3), got {arr.shape}")
        if len(arr) == 0:
            raise ValueError(f"{name} is empty")
        if strict and len(arr) != EXPECTED[name]:
            raise ValueError(f"{name}: expected {EXPECTED[name]} windows, got {len(arr)}")
        arrays[name] = np.asarray(arr, dtype=np.float32)
    return arrays


def build_model():
    import tensorflow as tf

    x = tf.keras.Input(shape=(16, 3), name="raw_standardized_window")
    h = tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu")(x)
    h = tf.keras.layers.Conv1D(8, 3, strides=2, padding="same", activation="relu")(h)
    h = tf.keras.layers.Conv1D(4, 3, strides=2, padding="same", activation="relu", name="bottleneck")(h)
    h = tf.keras.layers.UpSampling1D(2)(h)
    h = tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")(h)
    h = tf.keras.layers.UpSampling1D(2)(h)
    h = tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu")(h)
    output = tf.keras.layers.Conv1D(3, 3, padding="same", activation="linear")(h)
    model = tf.keras.Model(x, output, name="cnn_ae_16x3")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def errors(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    restored = model.predict(x, batch_size=64, verbose=0)
    per_channel = np.mean(np.square(x.astype(np.float64) - restored.astype(np.float64)), axis=1)
    return per_channel.mean(axis=1), per_channel


def calibration_threshold(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    """Finite-sample rank threshold: score > threshold means anomalous.

    If ceil((n+1)*(1-alpha)) > n, use +inf (no nontrivial bound at that alpha).
    The false alarm bound assumes exchangeable future normal scores.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return (float(np.sort(scores)[rank - 1]) if rank <= n else float("inf")), rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Validated cnn_ae_data.npz")
    parser.add_argument("--out", type=Path, default=Path("cnn_ae_results"))
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--allow-other-counts", action="store_true", help="Accept other N while keeping 16x3")
    parser.add_argument("--alpha", type=float, default=0.05, help="Target calibration false alarm bound")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data = load_data(args.data, strict=not args.allow_other_counts)
    print("Input arrays:", {k: list(v.shape) for k, v in data.items()})
    if args.check_data:
        return

    # Set before importing TensorFlow for repeatability where the backend permits.
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    import tensorflow as tf
    from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
)

    tf.keras.utils.set_random_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    model = build_model()
    history = model.fit(
        data["X_train"], data["X_train"],
        validation_data=(data["X_val"], data["X_val"]),
        epochs=args.epochs, batch_size=args.batch_size, shuffle=True, verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25,
                                             restore_best_weights=True, min_delta=1e-5),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                 patience=10, min_lr=1e-5),
        ],
    )
    # Set threshold with normal calibration only. Never adjust it after inspecting test labels.
    cal_score, _ = errors(model, data["X_cal"])
    threshold, rank = calibration_threshold(cal_score, args.alpha)
    normal_score, normal_channels = errors(model, data["X_test_normal"])
    anomaly_score, anomaly_channels = errors(model, data["X_test_anomaly"])
    score = np.r_[normal_score, anomaly_score]
    truth = np.r_[np.zeros(len(normal_score), dtype=int), np.ones(len(anomaly_score), dtype=int)]
    prediction = (score > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel().tolist()
    result = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "f1_score": float(f1_score(truth, prediction)),
        "recall": float(recall_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "specificity": tn / (tn + fp),
        "model": "1D CNN autoencoder, bottleneck (4,4), MSE across 16 timesteps and 3 channels",
        "source_npz": str(args.data.resolve()), "seed": args.seed,
        "alpha": args.alpha, "calibration_count": len(cal_score),
        "threshold_rank_1_based": rank, "threshold": threshold,
        "decision_rule": "reconstruction_mse > threshold",
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "normal_test_count": len(normal_score), "anomaly_test_count": len(anomaly_score),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "false_positive_rate": fp / (fp + tn), "anomaly_recall": tp / (tp + fn),
        "roc_auc": float(roc_auc_score(truth, score)),
        "average_precision": float(average_precision_score(truth, score)),
        "channel_names": CHANNELS,
        "channel_mean_mse_normal": normal_channels.mean(axis=0).tolist(),
        "channel_mean_mse_anomaly": anomaly_channels.mean(axis=0).tolist(),
        "limitations": "Normal/anomaly collected on different days; 28 anomaly windows are not 28 independent failures. Calibration bound needs exchangeable normal data.",
    }
    model.save(args.out / "cnn_ae.keras")
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (args.out / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    np.savez_compressed(args.out / "scores.npz", calibration_normal=cal_score,
                        test_normal=normal_score, test_anomaly=anomaly_score,
                        test_normal_per_channel=normal_channels,
                        test_anomaly_per_channel=anomaly_channels)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
