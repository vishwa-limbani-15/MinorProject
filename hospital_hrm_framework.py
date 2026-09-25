"""
AI-Driven Hospital HRM Framework
================================

This script provides a detailed reference implementation of the proposed
AI-driven Human Resource Management (HRM) framework for hospital workforce
planning, scheduling, and performance evaluation.

It includes:
1. Workforce demand forecasting (LSTM, XGBoost, Random Forest)
2. Intelligent staff scheduling (mixed-integer style optimization with OR-Tools)
3. Performance evaluation (structured KPIs + NLP sentiment analysis)
4. End-to-end orchestration utilities

The code is written as a research-grade prototype. It is modular, extensible,
and designed to work with CSV data similar to the synthetic dataset used in the
paper.

Author note:
- This is a detailed implementation template, not a hospital-certified product.
- Replace file paths and tune constraints based on your real deployment needs.
"""

from __future__ import annotations

import os
import math
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

from ortools.sat.python import cp_model

from transformers import pipeline as hf_pipeline

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("hospital_hrm")


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------
@dataclass
class ForecastConfig:
    target_col: str = "admissions"
    date_col: str = "date"
    unit_col: str = "unit"
    sequence_length: int = 28
    horizon: int = 14
    test_ratio: float = 0.30
    random_state: int = 42

    # LSTM defaults
    lstm_units_1: int = 64
    lstm_units_2: int = 32
    dropout: float = 0.20
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10


@dataclass
class SchedulingConfig:
    max_solver_seconds: int = 120
    fairness_weight: int = 3
    preference_weight: int = 2
    uncovered_weight: int = 100
    late_shift_penalty: int = 1

    # Labor rules (example defaults)
    max_hours_per_week: int = 48
    min_rest_hours: int = 10

    # Shift durations (hours)
    shift_hours: Dict[str, int] = field(
        default_factory=lambda: {
            "day": 8,
            "evening": 8,
            "night": 8,
        }
    )

    # Forbidden consecutive patterns (example)
    forbidden_pairs: List[Tuple[str, str]] = field(
        default_factory=lambda: [("night", "day")]
    )


@dataclass
class EvaluationConfig:
    text_batch_size: int = 32
    sentiment_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    positive_threshold: float = 0.60
    negative_threshold: float = 0.40


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def safe_divide(a: float, b: float) -> float:
    """Safely divide two numbers."""
    return a / b if b not in (0, 0.0) else 0.0


def compute_gini(values: Iterable[float]) -> float:
    """
    Compute the Gini coefficient for workload fairness.

    Parameters
    ----------
    values : Iterable[float]
        Workload values (e.g., assigned hours per staff member).

    Returns
    -------
    float
        Gini coefficient in [0, 1]. Lower is fairer.
    """
    arr = np.array(list(values), dtype=float)
    if len(arr) == 0:
        return 0.0
    if np.all(arr == 0):
        return 0.0

    arr = np.sort(arr)
    n = len(arr)
    cumulative = np.cumsum(arr)
    numerator = 2 * np.sum((np.arange(1, n + 1) * arr))
    denominator = n * np.sum(arr)
    gini = numerator / denominator - (n + 1) / n
    return float(max(0.0, min(1.0, gini)))


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return regression metrics used in the paper."""
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)),
    }


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------
class HospitalDataManager:
    """
    Handles loading, validation, preprocessing, and feature engineering for the
    hospital HRM framework.
    """

    def __init__(self, base_path: str):
        self.base_path = base_path

    def load_csv(self, name: str) -> pd.DataFrame:
        path = os.path.join(self.base_path, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path)
        logger.info("Loaded %s with shape %s", name, df.shape)
        return df

    def load_forecasting_data(self, filename: str = "admissions_daily.csv") -> pd.DataFrame:
        df = self.load_csv(filename)
        required_cols = {"date", "unit", "admissions", "census"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Forecasting data missing columns: {missing}")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["unit", "date"]).reset_index(drop=True)

        # Feature engineering aligned with paper
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["month"] = df["date"].dt.month
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

        # Lag and rolling features per unit
        for lag in [1, 7, 14]:
            df[f"admissions_lag_{lag}"] = df.groupby("unit")["admissions"].shift(lag)
            df[f"census_lag_{lag}"] = df.groupby("unit")["census"].shift(lag)

        for window in [7, 14]:
            df[f"admissions_roll_mean_{window}"] = (
                df.groupby("unit")["admissions"]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
            df[f"admissions_roll_std_{window}"] = (
                df.groupby("unit")["admissions"]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).std())
            )

        df = df.dropna().reset_index(drop=True)
        return df

    def load_scheduling_inputs(
        self,
        staff_file: str = "staff.csv",
        demand_file: str = "demand_shift.csv",
        availability_file: str = "availability.csv",
        preferences_file: str = "preferences.csv",
        staff_skills_file: str = "staff_skills.csv",
    ) -> Dict[str, pd.DataFrame]:
        return {
            "staff": self.load_csv(staff_file),
            "demand": self.load_csv(demand_file),
            "availability": self.load_csv(availability_file),
            "preferences": self.load_csv(preferences_file),
            "staff_skills": self.load_csv(staff_skills_file),
        }

    def load_evaluation_inputs(
        self,
        attendance_file: str = "attendance_executed.csv",
        kpi_file: str = "staff_week_kpi.csv",
        patient_feedback_file: str = "patient_feedback.csv",
        peer_reviews_file: str = "peer_reviews.csv",
    ) -> Dict[str, pd.DataFrame]:
        return {
            "attendance": self.load_csv(attendance_file),
            "kpi": self.load_csv(kpi_file),
            "patient_feedback": self.load_csv(patient_feedback_file),
            "peer_reviews": self.load_csv(peer_reviews_file),
        }


# -----------------------------------------------------------------------------
# Forecasting module
# -----------------------------------------------------------------------------
class WorkforceDemandForecaster:
    """
    Implements the demand forecasting component using:
    - LSTM for temporal sequence modeling
    - XGBoost for nonlinear tree-based regression
    - Random Forest as a strong baseline
    """

    def __init__(self, config: ForecastConfig):
        self.config = config
        self.best_tabular_model = None
        self.lstm_model = None
        self.feature_columns: List[str] = []
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def _prepare_tabular_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(len(df) * (1 - self.config.test_ratio))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        logger.info("Tabular split complete: train=%s test=%s", train_df.shape, test_df.shape)
        return train_df, test_df

    def train_tree_models(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Train XGBoost and Random Forest on engineered tabular features.
        Returns performance metrics for each model.
        """
        excluded = {self.config.date_col, self.config.target_col}
        self.feature_columns = [c for c in df.columns if c not in excluded]

        # Encode unit as category codes if still textual
        work_df = df.copy()
        if work_df[self.config.unit_col].dtype == object:
            work_df[self.config.unit_col] = work_df[self.config.unit_col].astype("category").cat.codes

        train_df, test_df = self._prepare_tabular_split(work_df)
        X_train = train_df[self.feature_columns]
        y_train = train_df[self.config.target_col]
        X_test = test_df[self.feature_columns]
        y_test = test_df[self.config.target_col]

        models = {
            "XGBoost": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=500,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective="reg:squarederror",
                            random_state=self.config.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "RandomForest": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=300,
                            max_depth=12,
                            min_samples_split=4,
                            random_state=self.config.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        }

        results: Dict[str, Dict[str, float]] = {}
        best_mae = float("inf")

        for name, model in models.items():
            logger.info("Training %s", name)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate_regression(y_test.values, preds)
            results[name] = metrics
            logger.info("%s metrics: %s", name, metrics)

            if metrics["MAE"] < best_mae:
                best_mae = metrics["MAE"]
                self.best_tabular_model = model

        return results

    def _build_lstm_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build LSTM sequences using a sliding window over admissions and census.
        This follows the methodological idea of using the previous 28 days to
        predict the next demand value.
        """
        seq_len = self.config.sequence_length
        feature_cols = ["admissions", "census", "day_of_week", "is_weekend", "month"]

        sequences_X: List[np.ndarray] = []
        sequences_y: List[float] = []

        # Build per unit to preserve temporal structure
        for unit, g in df.groupby("unit"):
            g = g.sort_values("date").reset_index(drop=True)
            unit_features = g[feature_cols].astype(float).values
            unit_target = g[self.config.target_col].astype(float).values

            for i in range(seq_len, len(g)):
                sequences_X.append(unit_features[i - seq_len : i])
                sequences_y.append(unit_target[i])

        X = np.array(sequences_X, dtype=np.float32)
        y = np.array(sequences_y, dtype=np.float32).reshape(-1, 1)

        split_idx = int(len(X) * (1 - self.config.test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale target only; sequence features stay in numeric range already,
        # but we normalize them for training stability.
        n_train, seq, feat = X_train.shape
        X_train_2d = X_train.reshape(-1, feat)
        X_test_2d = X_test.reshape(-1, feat)

        self.scaler_X.fit(X_train_2d)
        X_train_scaled = self.scaler_X.transform(X_train_2d).reshape(n_train, seq, feat)
        X_test_scaled = self.scaler_X.transform(X_test_2d).reshape(X_test.shape[0], seq, feat)

        self.scaler_y.fit(y_train)
        y_train_scaled = self.scaler_y.transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def _build_lstm_model(self, n_features: int) -> tf.keras.Model:
        model = Sequential(
            [
                LSTM(self.config.lstm_units_1, return_sequences=True, input_shape=(self.config.sequence_length, n_features)),
                Dropout(self.config.dropout),
                LSTM(self.config.lstm_units_2),
                Dropout(self.config.dropout),
                Dense(1, activation="linear"),
            ]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss="mae")
        return model

    def train_lstm(self, df: pd.DataFrame) -> Dict[str, float]:
        logger.info("Preparing LSTM sequences")
        X_train, X_test, y_train, y_test = self._build_lstm_sequences(df)

        logger.info("Training LSTM with X_train=%s X_test=%s", X_train.shape, X_test.shape)
        self.lstm_model = self._build_lstm_model(n_features=X_train.shape[2])
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.config.patience,
            restore_best_weights=True,
        )

        self.lstm_model.fit(
            X_train,
            y_train,
            validation_split=0.1,
            epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            callbacks=[early_stopping],
            verbose=0,
        )

        pred_scaled = self.lstm_model.predict(X_test, verbose=0)
        y_pred = self.scaler_y.inverse_transform(pred_scaled).flatten()
        y_true = self.scaler_y.inverse_transform(y_test).flatten()

        metrics = evaluate_regression(y_true, y_pred)
        logger.info("LSTM metrics: %s", metrics)
        return metrics

    def compare_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """Train all forecasting models and return a metrics table."""
        tree_results = self.train_tree_models(df)
        lstm_results = self.train_lstm(df)

        result_rows = [{"Model": "LSTM", **lstm_results}]
        for model_name, metrics in tree_results.items():
            result_rows.append({"Model": model_name, **metrics})

        results_df = pd.DataFrame(result_rows).sort_values("MAE").reset_index(drop=True)
        logger.info("Forecast model comparison complete")
        return results_df

    def forecast_future(self, recent_history: pd.DataFrame, model_type: str = "lstm") -> np.ndarray:
        """
        Forecast next-step demand from recent history.
        This is a simplified inference function for demonstration.
        """
        model_type = model_type.lower()

        if model_type == "lstm":
            if self.lstm_model is None:
                raise RuntimeError("LSTM model has not been trained.")
            if len(recent_history) < self.config.sequence_length:
                raise ValueError("Insufficient history for LSTM sequence prediction.")

            work = recent_history.tail(self.config.sequence_length).copy()
            feature_cols = ["admissions", "census", "day_of_week", "is_weekend", "month"]
            X = work[feature_cols].astype(float).values.reshape(1, self.config.sequence_length, len(feature_cols))
            X_2d = X.reshape(-1, len(feature_cols))
            X_scaled = self.scaler_X.transform(X_2d).reshape(1, self.config.sequence_length, len(feature_cols))
            pred_scaled = self.lstm_model.predict(X_scaled, verbose=0)
            pred = self.scaler_y.inverse_transform(pred_scaled)
            return pred.flatten()

        if self.best_tabular_model is None:
            raise RuntimeError("Tabular model has not been trained.")

        latest = recent_history.tail(1).copy()
        if latest[self.config.unit_col].dtype == object:
            latest[self.config.unit_col] = latest[self.config.unit_col].astype("category").cat.codes
        X = latest[self.feature_columns]
        return self.best_tabular_model.predict(X)


# -----------------------------------------------------------------------------
# Scheduling module
# -----------------------------------------------------------------------------
class IntelligentStaffScheduler:
    """
    Implements the scheduling module using OR-Tools CP-SAT.

    The objective approximates the paper's formulation by minimizing:
    - uncovered demand
    - preference dissatisfaction
    - workload imbalance (soft fairness)
    """

    def __init__(self, config: SchedulingConfig):
        self.config = config

    def _build_lookup_maps(
        self,
        staff_df: pd.DataFrame,
        availability_df: pd.DataFrame,
        preferences_df: pd.DataFrame,
        staff_skills_df: pd.DataFrame,
    ) -> Tuple[Dict, Dict, Dict]:
        """Create efficient lookup dictionaries."""
        availability_map = {
            (r.staff_id, r.date, r.shift): int(r.available)
            for r in availability_df.itertuples(index=False)
        }
        preference_map = {
            (r.staff_id, r.date, r.shift, r.unit): float(r.preference_score)
            for r in preferences_df.itertuples(index=False)
        }
        skill_map: Dict[str, set] = {}
        for row in staff_skills_df.itertuples(index=False):
            skill_map.setdefault(row.staff_id, set()).add(row.skill)

        return availability_map, preference_map, skill_map

    def generate_weekly_roster(
        self,
        staff_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        availability_df: pd.DataFrame,
        preferences_df: pd.DataFrame,
        staff_skills_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Generate a weekly roster using CP-SAT.

        Expected columns:
        - staff_df: staff_id, contract_type, max_hours_week
        - demand_df: date, unit, shift, required_staff, required_skill (optional)
        - availability_df: staff_id, date, shift, available
        - preferences_df: staff_id, date, shift, unit, preference_score
        - staff_skills_df: staff_id, skill
        """
        model = cp_model.CpModel()

        # Normalize dtypes to strings for stable keying
        for df in [staff_df, demand_df, availability_df, preferences_df, staff_skills_df]:
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str)

        availability_map, preference_map, skill_map = self._build_lookup_maps(
            staff_df, availability_df, preferences_df, staff_skills_df
        )

        staff_ids = staff_df["staff_id"].astype(str).tolist()
        demand_rows = list(demand_df.itertuples(index=False))

        # Decision variables x[(staff, demand_idx)] = assigned or not
        x: Dict[Tuple[str, int], cp_model.IntVar] = {}
        uncovered: Dict[int, cp_model.IntVar] = {}

        for d_idx, row in enumerate(demand_rows):
            uncovered[d_idx] = model.NewIntVar(0, int(row.required_staff), f"uncovered_{d_idx}")
            for sid in staff_ids:
                x[(sid, d_idx)] = model.NewBoolVar(f"x_{sid}_{d_idx}")

        # Constraint 1: Coverage with slack
        for d_idx, row in enumerate(demand_rows):
            assigned_vars = [x[(sid, d_idx)] for sid in staff_ids]
            model.Add(sum(assigned_vars) + uncovered[d_idx] >= int(row.required_staff))

        # Constraint 2: Availability
        for d_idx, row in enumerate(demand_rows):
            for sid in staff_ids:
                available = availability_map.get((sid, str(row.date), str(row.shift)), 0)
                if available == 0:
                    model.Add(x[(sid, d_idx)] == 0)

        # Constraint 3: Skill matching (if required_skill exists and not null)
        has_required_skill_col = "required_skill" in demand_df.columns
        if has_required_skill_col:
            for d_idx, row in enumerate(demand_rows):
                req_skill = getattr(row, "required_skill", None)
                if req_skill and str(req_skill).lower() not in {"", "none", "nan"}:
                    for sid in staff_ids:
                        if req_skill not in skill_map.get(sid, set()):
                            model.Add(x[(sid, d_idx)] == 0)

        # Constraint 4: Single assignment per staff/date/shift across units
        demand_key_map: Dict[Tuple[str, str], List[int]] = {}
        for d_idx, row in enumerate(demand_rows):
            demand_key_map.setdefault((str(row.date), str(row.shift)), []).append(d_idx)

        for sid in staff_ids:
            for key, indices in demand_key_map.items():
                model.Add(sum(x[(sid, i)] for i in indices) <= 1)

        # Constraint 5: Weekly max hours
        max_hours_map = {
            str(r.staff_id): int(getattr(r, "max_hours_week", self.config.max_hours_per_week))
            for r in staff_df.itertuples(index=False)
        }
        shift_hour_map = self.config.shift_hours

        for sid in staff_ids:
            workload_terms = []
            for d_idx, row in enumerate(demand_rows):
                hours = shift_hour_map.get(str(row.shift), 8)
                workload_terms.append(x[(sid, d_idx)] * hours)
            model.Add(sum(workload_terms) <= max_hours_map.get(sid, self.config.max_hours_per_week))

        # Constraint 6: Forbidden consecutive pairs (e.g., night -> day)
        # Simplified implementation by checking adjacent dates in demand rows.
        demand_meta = pd.DataFrame(
            [{
                "d_idx": i,
                "date": pd.to_datetime(r.date),
                "shift": str(r.shift),
            } for i, r in enumerate(demand_rows)]
        )

        for sid in staff_ids:
            for shift_a, shift_b in self.config.forbidden_pairs:
                df_a = demand_meta[demand_meta["shift"] == shift_a]
                df_b = demand_meta[demand_meta["shift"] == shift_b]
                for ra in df_a.itertuples(index=False):
                    next_day = ra.date + pd.Timedelta(days=1)
                    candidates = df_b[df_b["date"] == next_day]
                    for rb in candidates.itertuples(index=False):
                        model.Add(x[(sid, int(ra.d_idx))] + x[(sid, int(rb.d_idx))] <= 1)

        # Soft fairness approximation: minimize absolute deviation from mean assignments
        staff_assignment_count: Dict[str, cp_model.IntVar] = {}
        total_possible_assignments = len(demand_rows)
        max_assignments = total_possible_assignments
        for sid in staff_ids:
            count_var = model.NewIntVar(0, max_assignments, f"assign_count_{sid}")
            model.Add(count_var == sum(x[(sid, d_idx)] for d_idx in range(len(demand_rows))))
            staff_assignment_count[sid] = count_var

        avg_assignments = max(1, len(demand_rows) // max(1, len(staff_ids)))
        fairness_deviation_vars = []
        for sid in staff_ids:
            dev = model.NewIntVar(0, max_assignments, f"fair_dev_{sid}")
            model.AddAbsEquality(dev, staff_assignment_count[sid] - avg_assignments)
            fairness_deviation_vars.append(dev)

        # Preference dissatisfaction
        preference_penalties = []
        for d_idx, row in enumerate(demand_rows):
            for sid in staff_ids:
                pref = preference_map.get((sid, str(row.date), str(row.shift), str(row.unit)), 0.5)
                # Convert [0,1] preference to integer penalty. Higher preference => lower penalty.
                penalty = int(round((1.0 - pref) * 10))
                if penalty > 0:
                    preference_penalties.append(x[(sid, d_idx)] * penalty)

        # Objective
        objective_terms = []
        objective_terms.extend(self.config.uncovered_weight * uncovered[i] for i in uncovered)
        objective_terms.extend(self.config.preference_weight * term for term in preference_penalties)
        objective_terms.extend(self.config.fairness_weight * dev for dev in fairness_deviation_vars)
        model.Minimize(sum(objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(self.config.max_solver_seconds)
        solver.parameters.num_search_workers = 8

        logger.info("Starting roster optimization with %d staff and %d demand rows", len(staff_ids), len(demand_rows))
        status = solver.Solve(model)
        logger.info("Solver status: %s", solver.StatusName(status))

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError("No feasible roster found.")

        output_rows = []
        workload_hours: Dict[str, int] = {sid: 0 for sid in staff_ids}
        total_preferences = 0
        satisfied_preferences = 0
        total_uncovered = 0

        for d_idx, row in enumerate(demand_rows):
            row_assigned = 0
            for sid in staff_ids:
                if solver.Value(x[(sid, d_idx)]) == 1:
                    pref = preference_map.get((sid, str(row.date), str(row.shift), str(row.unit)), 0.5)
                    total_preferences += 1
                    if pref >= 0.6:
                        satisfied_preferences += 1

                    hours = self.config.shift_hours.get(str(row.shift), 8)
                    workload_hours[sid] += hours
                    row_assigned += 1
                    output_rows.append(
                        {
                            "staff_id": sid,
                            "date": str(row.date),
                            "unit": str(row.unit),
                            "shift": str(row.shift),
                            "assigned": 1,
                            "preference_score": pref,
                            "shift_hours": hours,
                        }
                    )

            total_uncovered += max(0, int(row.required_staff) - row_assigned)

        roster_df = pd.DataFrame(output_rows)
        metrics = {
            "solver_status": solver.StatusName(status),
            "uncovered_positions": float(total_uncovered),
            "preference_satisfaction": safe_divide(satisfied_preferences, total_preferences) * 100.0,
            "workload_gini": compute_gini(workload_hours.values()),
            "solve_time_seconds": float(solver.WallTime()),
        }

        return roster_df, metrics


# -----------------------------------------------------------------------------
# Performance evaluation module
# -----------------------------------------------------------------------------
class PerformanceEvaluator:
    """
    Combines structured KPIs with NLP-based sentiment analysis to create a
    transparent staff performance evaluation layer.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.sentiment_pipe = None

    def _load_sentiment_pipeline(self):
        if self.sentiment_pipe is None:
            logger.info("Loading sentiment model: %s", self.config.sentiment_model_name)
            self.sentiment_pipe = hf_pipeline(
                "sentiment-analysis",
                model=self.config.sentiment_model_name,
                tokenizer=self.config.sentiment_model_name,
                truncation=True,
            )

    def classify_sentiment(self, texts: List[str]) -> List[str]:
        """
        Convert transformer outputs into paper-style labels:
        positive / neutral / negative.
        """
        self._load_sentiment_pipeline()
        predictions = self.sentiment_pipe(texts, batch_size=self.config.text_batch_size)

        labels: List[str] = []
        for pred in predictions:
            model_label = str(pred["label"]).upper()
            score = float(pred["score"])

            # DistilBERT SST-2 returns POSITIVE/NEGATIVE. We map low-confidence
            # predictions into a neutral class for the research framework.
            if model_label == "POSITIVE":
                if score >= self.config.positive_threshold:
                    labels.append("positive")
                else:
                    labels.append("neutral")
            else:
                if score >= self.config.positive_threshold:
                    labels.append("negative")
                else:
                    labels.append("neutral")

        return labels

    def evaluate_structured_kpis(
        self,
        attendance_df: pd.DataFrame,
        kpi_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Compute aggregated structured performance indicators.
        Expected examples:
        - attendance_df: punctual (0/1), absent (0/1)
        - kpi_df: task_completion_rate, response_time_minutes
        """
        metrics: Dict[str, float] = {}

        if "task_completion_rate" in kpi_df.columns:
            metrics["task_completion_avg"] = float(kpi_df["task_completion_rate"].mean())

        if "response_time_minutes" in kpi_df.columns:
            metrics["response_time_avg"] = float(kpi_df["response_time_minutes"].mean())

        if "punctual" in attendance_df.columns:
            metrics["punctuality_rate"] = float(attendance_df["punctual"].mean())

        if "absent" in attendance_df.columns:
            metrics["absenteeism_rate"] = float(attendance_df["absent"].mean())

        return metrics

    def evaluate_feedback_sentiment(
        self,
        feedback_df: pd.DataFrame,
        text_col: str = "comment",
        department_col: str = "department",
        label_col: Optional[str] = "gold_sentiment",
    ) -> Dict[str, pd.DataFrame]:
        """
        Run sentiment inference and optionally evaluate against labeled subset.
        """
        if text_col not in feedback_df.columns:
            raise ValueError(f"Feedback dataframe must contain '{text_col}' column.")

        work = feedback_df.copy()
        work[text_col] = work[text_col].fillna("").astype(str)

        inferred_labels = self.classify_sentiment(work[text_col].tolist())
        work["predicted_sentiment"] = inferred_labels

        # Overall sentiment distribution
        sentiment_distribution = (
            work["predicted_sentiment"]
            .value_counts(normalize=True)
            .rename_axis("sentiment")
            .reset_index(name="proportion")
        )

        # Department-level aggregation if column exists
        if department_col in work.columns:
            dept_distribution = (
                work.groupby([department_col, "predicted_sentiment"]).size()
                .groupby(level=0)
                .apply(lambda s: s / s.sum())
                .reset_index(name="proportion")
            )
        else:
            dept_distribution = pd.DataFrame()

        outputs: Dict[str, pd.DataFrame] = {
            "feedback_with_predictions": work,
            "sentiment_distribution": sentiment_distribution,
            "department_sentiment_distribution": dept_distribution,
        }

        # Optional evaluation subset
        if label_col and label_col in work.columns:
            eval_df = work[work[label_col].notna()].copy()
            if not eval_df.empty:
                y_true = eval_df[label_col].astype(str).tolist()
                y_pred = eval_df["predicted_sentiment"].astype(str).tolist()
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
                acc = accuracy_score(y_true, y_pred)
                outputs["classifier_metrics"] = pd.DataFrame(
                    [
                        {
                            "accuracy": acc,
                            "precision_macro": precision,
                            "recall_macro": recall,
                            "f1_macro": f1,
                        }
                    ]
                )

        return outputs

    def build_department_summary(
        self,
        attendance_df: pd.DataFrame,
        kpi_df: pd.DataFrame,
        feedback_with_predictions: pd.DataFrame,
        department_col: str = "department",
    ) -> pd.DataFrame:
        """
        Build department-level summary integrating structured and unstructured data.
        """
        summaries = []

        all_departments = set()
        if department_col in attendance_df.columns:
            all_departments.update(attendance_df[department_col].dropna().astype(str).tolist())
        if department_col in kpi_df.columns:
            all_departments.update(kpi_df[department_col].dropna().astype(str).tolist())
        if department_col in feedback_with_predictions.columns:
            all_departments.update(feedback_with_predictions[department_col].dropna().astype(str).tolist())

        for dept in sorted(all_departments):
            att_sub = attendance_df[attendance_df.get(department_col, "") == dept].copy() if department_col in attendance_df.columns else pd.DataFrame()
            kpi_sub = kpi_df[kpi_df.get(department_col, "") == dept].copy() if department_col in kpi_df.columns else pd.DataFrame()
            fb_sub = feedback_with_predictions[feedback_with_predictions.get(department_col, "") == dept].copy() if department_col in feedback_with_predictions.columns else pd.DataFrame()

            row = {"department": dept}
            if not kpi_sub.empty and "task_completion_rate" in kpi_sub.columns:
                row["task_completion_avg"] = float(kpi_sub["task_completion_rate"].mean())
            if not att_sub.empty and "punctual" in att_sub.columns:
                row["punctuality_rate"] = float(att_sub["punctual"].mean())
            if not att_sub.empty and "absent" in att_sub.columns:
                row["absenteeism_rate"] = float(att_sub["absent"].mean())

            if not fb_sub.empty:
                for sentiment in ["positive", "neutral", "negative"]:
                    row[f"{sentiment}_feedback_ratio"] = float((fb_sub["predicted_sentiment"] == sentiment).mean())

            summaries.append(row)

        return pd.DataFrame(summaries)


# -----------------------------------------------------------------------------
# End-to-end orchestrator
# -----------------------------------------------------------------------------
class HospitalHRMFramework:
    """
    End-to-end orchestrator for the proposed AI-driven hospital HRM system.
    """

    def __init__(
        self,
        data_path: str,
        forecast_config: Optional[ForecastConfig] = None,
        scheduling_config: Optional[SchedulingConfig] = None,
        evaluation_config: Optional[EvaluationConfig] = None,
    ):
        self.data_manager = HospitalDataManager(data_path)
        self.forecaster = WorkforceDemandForecaster(forecast_config or ForecastConfig())
        self.scheduler = IntelligentStaffScheduler(scheduling_config or SchedulingConfig())
        self.evaluator = PerformanceEvaluator(evaluation_config or EvaluationConfig())

    def run_forecasting_experiment(self) -> pd.DataFrame:
        """Train and compare forecasting models."""
        forecast_df = self.data_manager.load_forecasting_data()
        results = self.forecaster.compare_models(forecast_df)
        logger.info("Forecasting experiment completed")
        return results

    def run_scheduling_experiment(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Generate a weekly roster and compute scheduling metrics."""
        inputs = self.data_manager.load_scheduling_inputs()
        roster_df, metrics = self.scheduler.generate_weekly_roster(
            staff_df=inputs["staff"],
            demand_df=inputs["demand"],
            availability_df=inputs["availability"],
            preferences_df=inputs["preferences"],
            staff_skills_df=inputs["staff_skills"],
        )
        logger.info("Scheduling experiment completed: %s", metrics)
        return roster_df, metrics

    def run_performance_evaluation(self) -> Dict[str, object]:
        """Compute structured metrics and NLP sentiment analytics."""
        inputs = self.data_manager.load_evaluation_inputs()

        structured_metrics = self.evaluator.evaluate_structured_kpis(
            attendance_df=inputs["attendance"],
            kpi_df=inputs["kpi"],
        )

        sentiment_outputs = self.evaluator.evaluate_feedback_sentiment(
            feedback_df=inputs["patient_feedback"],
            text_col="comment",
            department_col="department",
            label_col="gold_sentiment" if "gold_sentiment" in inputs["patient_feedback"].columns else None,
        )

        department_summary = self.evaluator.build_department_summary(
            attendance_df=inputs["attendance"],
            kpi_df=inputs["kpi"],
            feedback_with_predictions=sentiment_outputs["feedback_with_predictions"],
            department_col="department",
        )

        logger.info("Performance evaluation completed")
        return {
            "structured_metrics": structured_metrics,
            "sentiment_distribution": sentiment_outputs["sentiment_distribution"],
            "department_sentiment_distribution": sentiment_outputs["department_sentiment_distribution"],
            "classifier_metrics": sentiment_outputs.get("classifier_metrics", pd.DataFrame()),
            "department_summary": department_summary,
        }

    def export_outputs(self, output_dir: str) -> Dict[str, str]:
        """
        Run all core modules and export outputs to CSV/JSON files.
        """
        os.makedirs(output_dir, exist_ok=True)
        exported: Dict[str, str] = {}

        # Forecasting
        forecast_results = self.run_forecasting_experiment()
        forecast_path = os.path.join(output_dir, "forecast_model_comparison.csv")
        forecast_results.to_csv(forecast_path, index=False)
        exported["forecast_model_comparison"] = forecast_path

        # Scheduling
        roster_df, scheduling_metrics = self.run_scheduling_experiment()
        roster_path = os.path.join(output_dir, "generated_weekly_roster.csv")
        metrics_path = os.path.join(output_dir, "scheduling_metrics.json")
        roster_df.to_csv(roster_path, index=False)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(scheduling_metrics, f, indent=2)
        exported["generated_weekly_roster"] = roster_path
        exported["scheduling_metrics"] = metrics_path

        # Performance evaluation
        perf = self.run_performance_evaluation()
        perf_structured_path = os.path.join(output_dir, "performance_structured_metrics.json")
        with open(perf_structured_path, "w", encoding="utf-8") as f:
            json.dump(perf["structured_metrics"], f, indent=2)
        exported["performance_structured_metrics"] = perf_structured_path

        for key in [
            "sentiment_distribution",
            "department_sentiment_distribution",
            "classifier_metrics",
            "department_summary",
        ]:
            df = perf[key]
            if isinstance(df, pd.DataFrame) and not df.empty:
                p = os.path.join(output_dir, f"{key}.csv")
                df.to_csv(p, index=False)
                exported[key] = p

        logger.info("All outputs exported to %s", output_dir)
        return exported


# -----------------------------------------------------------------------------
# Example usage / entry point
# -----------------------------------------------------------------------------
def main():
    """
    Example execution.

    Expected directory structure (replace with your own path):
    data/
      admissions_daily.csv
      staff.csv
      demand_shift.csv
      availability.csv
      preferences.csv
      staff_skills.csv
      attendance_executed.csv
      staff_week_kpi.csv
      patient_feedback.csv
      peer_reviews.csv
    """
    data_path = "./data"
    output_path = "./outputs"

    framework = HospitalHRMFramework(data_path=data_path)

    try:
        exported = framework.export_outputs(output_dir=output_path)
        print("Export completed successfully. Files:")
        for name, path in exported.items():
            print(f"- {name}: {path}")
    except Exception as exc:
        logger.exception("Framework execution failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
