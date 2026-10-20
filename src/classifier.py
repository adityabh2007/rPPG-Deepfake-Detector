#Build a video dataset, compare classifiers, and save an SVM model.

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from src.extract_signal import extract_rgb_traces, pos_algorithm
from src.features import compute_features
from src.filters import bandpass, detrend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_PATH = RESULTS_DIR / "features.npz"
MODEL_PATH = RESULTS_DIR / "model.joblib"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
FEATURE_COUNT = 21
RANDOM_SEED = 42
CLASS_LABELS = {"fake": 0, "real": 1}


def _list_videos(real_dir: str | Path, fake_dir: str | Path) -> list[Path]:
    videos: list[Path] = []
    for directory in (Path(real_dir), Path(fake_dir)):
        if directory.exists():
            videos.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            )
    return sorted(videos, key=lambda path: str(path).lower())


def _class_for_video(
    video_path: Path, real_dir: str | Path, fake_dir: str | Path
) -> str:
    resolved_path = video_path.resolve()
    real_root = Path(real_dir).resolve()
    fake_root = Path(fake_dir).resolve()
    if real_root == resolved_path.parent or real_root in resolved_path.parents:
        return "real"
    if fake_root == resolved_path.parent or fake_root in resolved_path.parents:
        return "fake"
    raise ValueError(f"{video_path} is not inside either class folder.")


def _source_signature(paths: list[Path]) -> NDArray[np.str_]:
    signatures = []
    for path in paths:
        stat = path.stat()
        signatures.append(f"{stat.st_size}:{stat.st_mtime_ns}")
    return np.asarray(signatures, dtype=np.str_)


def extract_video_features(
    video_path: str | Path,
) -> tuple[NDArray[np.float64], list[str]]:
    traces, fps = extract_rgb_traces(video_path)
    pulse_signals = {}
    for roi_name, rgb_trace in traces.items():
        pulse = pos_algorithm(rgb_trace, fps)
        pulse_signals[roi_name] = bandpass(detrend(pulse), fps)
    return compute_features(pulse_signals, fps)


def build_dataset(
    real_dir: str | Path, fake_dir: str | Path
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    video_paths = _list_videos(real_dir, fake_dir)
    source_paths = np.asarray([str(path.resolve()) for path in video_paths], dtype=np.str_)
    source_signatures = _source_signature(video_paths)

    if CACHE_PATH.exists():
        try:
            with np.load(CACHE_PATH, allow_pickle=False) as cached:
                cached_paths = cached["source_paths"]
                cached_signatures = cached["source_signatures"]
                if np.array_equal(cached_paths, source_paths) and np.array_equal(
                    cached_signatures, source_signatures
                ):
                    print(f"Using cached features from {CACHE_PATH}.")
                    return (
                        np.asarray(cached["X"], dtype=np.float64),
                        np.asarray(cached["y"], dtype=np.int64),
                        cached["video_names"].astype(str).tolist(),
                    )
        except (OSError, KeyError, ValueError) as error:
            print(f"Warning: could not read the feature cache ({error}); rebuilding it.")

    features: list[NDArray[np.float64]] = []
    labels: list[int] = []
    video_names: list[str] = []
    processed_paths: list[str] = []
    feature_names: list[str] | None = None

    if not video_paths:
        print("No supported video files found in the real and fake folders.")

    for video_path in tqdm(video_paths, desc="Extracting video features"):
        try:
            class_name = _class_for_video(video_path, real_dir, fake_dir)
            row, current_feature_names = extract_video_features(video_path)
            if feature_names is not None and current_feature_names != feature_names:
                raise ValueError("Feature names changed between videos.")
            feature_names = current_feature_names
            features.append(row)
            labels.append(CLASS_LABELS[class_name])
            video_names.append(f"{class_name}/{video_path.name}")
            processed_paths.append(str(video_path.resolve()))
        except Exception as error:
            print(f"Warning: skipping {video_path.name}: {error}")

    if features:
        X = np.vstack(features).astype(np.float64)
    else:
        X = np.empty((0, FEATURE_COUNT), dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)

    np.savez_compressed(
        CACHE_PATH,
        X=X,
        y=y,
        video_names=np.asarray(video_names, dtype=np.str_),
        processed_paths=np.asarray(processed_paths, dtype=np.str_),
        source_paths=source_paths,
        source_signatures=source_signatures,
        feature_names=np.asarray(feature_names or [], dtype=np.str_),
    )
    print(f"Saved {len(video_names)} video feature rows to {CACHE_PATH}.")
    return X, y, video_names


def _make_svm() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
            ),
        ]
    )


def train_and_evaluate(
    X: NDArray[np.float64], y: NDArray[np.int64]
) -> Pipeline:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be a feature matrix aligned with one-dimensional y.")
    if len(y) == 0 or len(np.unique(y)) < 2:
        raise ValueError("Training needs both real and fake videos.")
    class_counts = np.bincount(y)
    if len(class_counts) < 2 or np.min(class_counts[:2]) < 5:
        raise ValueError(
            "Stratified 5-fold evaluation needs at least 5 videos in each class."
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    svm_model = _make_svm()
    predicted = cross_val_predict(svm_model, X, y, cv=cv, method="predict")
    probabilities = cross_val_predict(svm_model, X, y, cv=cv, method="predict_proba")
    svm_class_order = np.unique(y)
    real_index = int(np.flatnonzero(svm_class_order == CLASS_LABELS["real"])[0])
    real_probabilities = probabilities[:, real_index]

    print("\nStratified 5-fold SVM evaluation:")
    print(f"  Accuracy:  {accuracy_score(y, predicted):.3f}")
    print(f"  Precision: {precision_score(y, predicted, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y, predicted, zero_division=0):.3f}")
    print(f"  F1:        {f1_score(y, predicted, zero_division=0):.3f}")
    print(f"  ROC-AUC:   {roc_auc_score(y, real_probabilities):.3f}")
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    print("  Confusion matrix (rows: true fake/real; columns: predicted fake/real):")
    print(matrix)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    false_positive_rate, true_positive_rate, _ = roc_curve(y, real_probabilities)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.plot(false_positive_rate, true_positive_rate, label="SVM ROC")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="5-fold ROC curve",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots()
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["fake", "real"])
    axis.set_yticks([0, 1], labels=["fake", "real"])
    axis.set(
        xlabel="Predicted label",
        ylabel="True label",
        title="5-fold confusion matrix",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    comparison_models: dict[str, Any] = {
        "SVM (RBF)": svm_model,
        "Logistic regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
            ]
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, class_weight="balanced"
        ),
    }
    print("\nModel comparison (5-fold accuracy):")
    for name, model in comparison_models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(f"  {name:<22} {scores.mean():.3f} +/- {scores.std():.3f}")

    final_model = _make_svm()
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)
    print(f"\nSaved final SVM model to {MODEL_PATH}.")
    return final_model


def predict_from_features(
    features: NDArray[np.float64],
) -> tuple[str, float]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run 'python main.py train' first."
        )
    model: Pipeline = joblib.load(MODEL_PATH)
    feature_row = np.asarray(features, dtype=np.float64).reshape(1, -1)
    probabilities = model.predict_proba(feature_row)[0]
    classes = model.classes_
    predicted_class = int(model.predict(feature_row)[0])
    predicted_index = int(np.flatnonzero(classes == predicted_class)[0])
    label = "real" if predicted_class == CLASS_LABELS["real"] else "fake"
    return label, float(probabilities[predicted_index])


def predict(video_path: str | Path) -> tuple[str, float]:
    features, _ = extract_video_features(video_path)
    return predict_from_features(features)
