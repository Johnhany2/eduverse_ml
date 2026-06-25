import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import pandas as pd


MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

# ── CHANGED: path to xlsx files (also inside models/) ────────────────────────
WEIGHTS_PATH = MODEL_DIR / "track_course_weights.xlsx"
TRACKS_PATH  = MODEL_DIR / "tracks.xlsx"
# ─────────────────────────────────────────────────────────────────────────────


class ModelLoadError(RuntimeError):
    pass


def classify_gba_rule(gba: float) -> str:
    if gba >= 3.5:
        return "Excellent"
    if gba >= 3.0:
        return "Very Good"
    if gba >= 2.5:
        return "Good"
    if gba >= 2.0:
        return "At Risk"
    return "Critical"


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _load_bundle(file_name: str) -> Dict[str, Any]:
    path = MODEL_DIR / file_name
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ModelLoadError(f"Invalid model bundle format: {file_name}")

    return bundle


def _dataframe_from_features(features: Iterable[str], values: Dict[str, Any]) -> pd.DataFrame:
    row = {feature: float(values.get(feature, 0) or 0) for feature in features}
    return pd.DataFrame([row], columns=list(features))


def _probability_map(bundle: Dict[str, Any], probabilities: Any) -> Dict[str, float]:
    model = bundle["model"]
    classes = getattr(model, "classes_", range(len(probabilities)))

    label_encoder = bundle.get("label_encoder")
    if label_encoder is not None:
        try:
            labels = label_encoder.inverse_transform(classes)
        except Exception:
            labels = label_encoder.classes_
    else:
        labels = classes

    return {
        str(_to_builtin(label)): round(float(prob), 6)
        for label, prob in zip(labels, probabilities)
    }


class ModelService:
    def __init__(self) -> None:
        self.risk  = _load_bundle("student_risk_model.pkl")
        self.gba   = _load_bundle("classfication_gba_model.pkl")
        self.track = _load_bundle("track_recommendation_model.pkl")

        # ── CHANGED: load xlsx reference tables once at startup ───────────────
        self._weights_df = pd.read_excel(WEIGHTS_PATH)
        self._tracks_df  = pd.read_excel(TRACKS_PATH)
        self._weights_df.columns = self._weights_df.columns.str.strip().str.lower()
        self._tracks_df.columns  = self._tracks_df.columns.str.strip().str.lower()
        # ─────────────────────────────────────────────────────────────────────

    def metadata(self) -> Dict[str, List[str]]:
        return {
            "risk_features":  list(self.risk["features"]),
            "gba_features":   list(self.gba["features"]),
            "track_features": list(self.track["features"]),
        }

    def predict_risk(self, values: Dict[str, Any]) -> Dict[str, Any]:
        features = self.risk["features"]
        frame = _dataframe_from_features(features, values)
        model = self.risk["model"]

        prediction = int(_to_builtin(model.predict(frame)[0]))
        probabilities: Dict[str, float] = {}
        risk_probability: Optional[float] = None

        if hasattr(model, "predict_proba"):
            raw_probabilities = model.predict_proba(frame)[0]
            probabilities = _probability_map(self.risk, raw_probabilities)
            risk_probability = probabilities.get("1")

        return {
            "at_risk":          prediction == 1,
            "prediction":       prediction,
            "risk_probability": risk_probability,
            "probabilities":    probabilities,
        }

    def predict_gba(self, values: Dict[str, Any]) -> Dict[str, Any]:
        features = self.gba["features"]
        frame = _dataframe_from_features(features, values)
        model = self.gba["model"]
        label_encoder = self.gba["label_encoder"]

        encoded_prediction = model.predict(frame)[0]
        model_prediction = str(label_encoder.inverse_transform([encoded_prediction])[0])

        probabilities: Dict[str, float] = {}
        if hasattr(model, "predict_proba"):
            raw_probabilities = model.predict_proba(frame)[0]
            probabilities = _probability_map(self.gba, raw_probabilities)

        rule_prediction = None
        current_gba = values.get("current_gba")
        if current_gba is not None:
            rule_prediction = classify_gba_rule(float(current_gba))

        return {
            "model_prediction": model_prediction,
            "rule_based_level": rule_prediction,
            "probabilities":    probabilities,
        }

    # ── CHANGED: accepts List[CourseGrade], computes track_scores internally ──
    def recommend_track(self, courses: List[Any]) -> Dict[str, Any]:
        # 1. Build grades DataFrame
        grades_df = pd.DataFrame(
            [{"course_id": c.course_id, "grade": c.grade} for c in courses]
        )

        # 2. Merge with weights → weighted_grade = grade * weight
        merged = pd.merge(
            grades_df,
            self._weights_df[["track_id", "course_id", "weight"]],
            on="course_id",
            how="inner",
        )

        # 3. Compute track scores: mean(weighted_grade) per track
        if merged.empty:
            track_scores: Dict[str, float] = {f: 0.0 for f in self.track["features"]}
        else:
            merged["weighted_grade"] = merged["grade"] * merged["weight"]
            track_scores = (
                merged.groupby("track_id")["weighted_grade"]
                .mean()
                .rename(lambda tid: f"track_{tid}_score")
                .to_dict()
            )

        # 4. Build feature row and predict (same logic as before)
        features = self.track["features"]
        frame = _dataframe_from_features(features, track_scores)
        model = self.track["model"]
        label_encoder = self.track["label_encoder"]

        encoded_prediction = model.predict(frame)[0]
        track_id = int(_to_builtin(label_encoder.inverse_transform([encoded_prediction])[0]))
        track_name = self._track_name(track_id) or str(track_id)

        probabilities = []
        if hasattr(model, "predict_proba"):
            raw_probabilities = model.predict_proba(frame)[0]
            classes = getattr(model, "classes_", range(len(raw_probabilities)))
            try:
                track_ids = label_encoder.inverse_transform(classes)
            except Exception:
                track_ids = label_encoder.classes_

            for current_track_id, probability in zip(track_ids, raw_probabilities):
                current_track_id = int(_to_builtin(current_track_id))
                probabilities.append(
                    {
                        "track_id":   current_track_id,
                        "track_name": self._track_name(current_track_id),
                        "probability": round(float(probability), 6),
                    }
                )

        return {
            "recommended_track_id":   track_id,
            "recommended_track_name": track_name,
            "probabilities":          probabilities,
        }
    # ─────────────────────────────────────────────────────────────────────────

    def _track_name(self, track_id: int) -> Optional[str]:
        # First try the xlsx table (more reliable)
        matched = self._tracks_df[self._tracks_df["track_id"] == track_id]
        if not matched.empty and "track_name" in matched.columns:
            return str(matched.iloc[0]["track_name"])

        # Fallback: tracks DataFrame saved inside the pkl bundle
        tracks = self.track.get("tracks")
        if tracks is None or "track_id" not in tracks.columns:
            return None
        matched = tracks[tracks["track_id"] == track_id]
        if matched.empty or "track_name" not in matched.columns:
            return None
        return str(matched.iloc[0]["track_name"])
import os
from pathlib import Path
from pyexpat import features, model
from typing import Any, Dict, Iterable, List, Optional

from pandas.core import frame

import joblib
import pandas as pd


MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))


class ModelLoadError(RuntimeError):
    pass


def classify_gba_rule(gba: float) -> str:
    if gba >= 3.5:
        return "Excellent"
    if gba >= 3.0:
        return "Very Good"
    if gba >= 2.5:
        return "Good"
    if gba >= 2.0:
        return "At Risk"
    return "Critical"


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _load_bundle(file_name: str) -> Dict[str, Any]:
    path = MODEL_DIR / file_name
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ModelLoadError(f"Invalid model bundle format: {file_name}")

    return bundle


def _dataframe_from_features(features: Iterable[str], values: Dict[str, Any]) -> pd.DataFrame:
    row = {feature: float(values.get(feature, 0) or 0) for feature in features}
    return pd.DataFrame([row], columns=list(features))


def _probability_map(bundle: Dict[str, Any], probabilities: Any) -> Dict[str, float]:
    model = bundle["model"]
    classes = getattr(model, "classes_", range(len(probabilities)))

    label_encoder = bundle.get("label_encoder")
    if label_encoder is not None:
        try:
            labels = label_encoder.inverse_transform(classes)
        except Exception:
            labels = label_encoder.classes_
    else:
        labels = classes

    return {
        str(_to_builtin(label)): round(float(prob), 6)
        for label, prob in zip(labels, probabilities)
    }


class ModelService:
    def __init__(self) -> None:
        self.risk = _load_bundle("student_risk_model.pkl")
        self.gba = _load_bundle("classfication_gba_model.pkl")
        self.track = _load_bundle("track_recommendation_model.pkl")
        self.track_weights = pd.read_excel(
            MODEL_DIR / "track_course_weights.xlsx"
)

    def metadata(self) -> Dict[str, List[str]]:
        return {
            "risk_features": list(self.risk["features"]),
            "gba_features": list(self.gba["features"]),
            "track_features": list(self.track["features"]),
        }

    def predict_risk(self, values: Dict[str, Any]) -> Dict[str, Any]:
        features = self.risk["features"]
        frame = _dataframe_from_features(features, values)
        model = self.risk["model"]

        prediction = int(_to_builtin(model.predict(frame)[0]))
        probabilities: Dict[str, float] = {}
        risk_probability: Optional[float] = None

        if hasattr(model, "predict_proba"):
            raw_probabilities = model.predict_proba(frame)[0]
            probabilities = _probability_map(self.risk, raw_probabilities)
            risk_probability = probabilities.get("1")

        return {
            "at_risk": prediction == 1,
            "prediction": prediction,
            "risk_probability": risk_probability,
            "probabilities": probabilities,
        }

    def predict_gba(self, values: Dict[str, Any]) -> Dict[str, Any]:
        features = self.gba["features"]
        frame = _dataframe_from_features(features, values)
        model = self.gba["model"]
        label_encoder = self.gba["label_encoder"]

        encoded_prediction = model.predict(frame)[0]
        model_prediction = str(label_encoder.inverse_transform([encoded_prediction])[0])

        probabilities: Dict[str, float] = {}
        if hasattr(model, "predict_proba"):
            raw_probabilities = model.predict_proba(frame)[0]
            probabilities = _probability_map(self.gba, raw_probabilities)

        rule_prediction = None
        current_gba = values.get("current_gba")
        if current_gba is not None:
            rule_prediction = classify_gba_rule(float(current_gba))

        return {
            "model_prediction": model_prediction,
            "rule_based_level": rule_prediction,
            "probabilities": probabilities,
        }

    def recommend_track(self, courses: List[Dict[str, Any]]) -> Dict[str, Any]:

        student_data = pd.DataFrame(courses)

        merged = student_data.merge(
            self.track_weights,
            on="course_id",
            how="inner"
        )

        merged["weighted_grade"] = (
            merged["grade"] * merged["weight"]
        )

        scores = merged.groupby("track_id")[
            "weighted_grade"
        ].mean()

        track_scores = {
            f"track_{track_id}_score": scores.get(track_id, 0)
            for track_id in self.track_weights["track_id"].unique()
        }

        features = self.track["features"]

        frame = _dataframe_from_features(
            features,
            track_scores
        )

        model = self.track["model"]

        label_encoder = self.track["label_encoder"]

        encoded_prediction = model.predict(frame)[0]

        track_id = int(
            _to_builtin(
                label_encoder.inverse_transform(
                    [encoded_prediction]
                )[0]
            )
        )

        track_name = self._track_name(track_id)

        probabilities = []

        if hasattr(model, "predict_proba"):

            raw_probabilities = model.predict_proba(frame)[0]

            classes = getattr(
                model,
                "classes_",
                range(len(raw_probabilities))
            )

            track_ids = label_encoder.inverse_transform(classes)

            for current_track_id, probability in zip(track_ids, raw_probabilities):

                current_track_id = int(_to_builtin(current_track_id))

                probabilities.append(
                    {
                        "track_id": current_track_id,
                        "track_name": self._track_name(current_track_id),
                        "probability": round(float(probability), 6),
                    }
                )

        return {
            "recommended_track_id": track_id,
            "recommended_track_name": track_name,
            "probabilities": probabilities,
        }



    def _track_name(self, track_id: int) -> Optional[str]:
        tracks = self.track.get("tracks")
        if tracks is None or "track_id" not in tracks.columns:
            return None

        matched = tracks[tracks["track_id"] == track_id]
        if matched.empty or "track_name" not in matched.columns:
            return None

        return str(matched.iloc[0]["track_name"])

