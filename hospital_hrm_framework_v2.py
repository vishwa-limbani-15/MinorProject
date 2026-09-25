"""
Hospital HRM Framework - Version 2
==================================

Version 2 keeps the original scheduling idea (CP-SAT optimization) but makes
it faithful to the supplied hospital_hrm_dataset.xlsx dataset.

Main upgrades over Version 1/original scheduler:
1. Excel workbook input instead of CSV-only input.
2. Correct dataset field mappings:
   - contracted_hours_per_week
   - skill_code
   - demand_shift min_* skill requirements
   - preference rows without a unit column
3. Skill-aware scheduling for multiple simultaneous skill requirements.
4. Separate skill-shortage slack so the model remains solvable when demand
   exceeds the available qualified workforce; unqualified staff are never
   counted toward a skill requirement.
5. Weekly contracted-hours constraints are applied per ISO week rather than
   incorrectly across the entire 56-day planning horizon.
6. Request-off preferences are treated as hard constraints.
7. One shift per staff member per day is enforced.
8. Detailed baseline evaluation is exported for later RL comparison.

This is a research prototype, not a clinical production scheduling system.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from ortools.sat.python import cp_model
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Version 2 requires OR-Tools. Install it with: pip install ortools"
    ) from exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("hospital_hrm_v2")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class V2SchedulingConfig:
    max_solver_seconds: int = 120
    num_search_workers: int = 8

    # Objective weights. Coverage/skill shortages dominate soft objectives.
    uncovered_weight: int = 1000
    skill_shortage_weight: int = 900
    preference_weight: int = 3
    fairness_weight: int = 2
    unnecessary_assignment_weight: int = 1

    # Shift durations in the supplied dataset.
    shift_hours: Dict[str, int] = field(
        default_factory=lambda: {
            "Day": 8,
            "Evening": 8,
            "Night": 8,
        }
    )

    # Research scheduling rule inherited from the original framework.
    max_hours_default: int = 40
    min_rest_hours: int = 10
    forbidden_pairs: List[Tuple[str, str]] = field(
        default_factory=lambda: [("Night", "Day")]
    )


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def safe_divide(a: float, b: float) -> float:
    return a / b if b not in (0, 0.0) else 0.0


def compute_gini(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0 or np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    total = arr.sum()
    return float(max(0.0, min(1.0, (2 * np.sum((np.arange(1, n + 1) * arr)) / (n * total)) - (n + 1) / n)))


def normalize_date(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def normalize_shift(value: str) -> str:
    value = str(value).strip()
    lookup = {"day": "Day", "evening": "Evening", "night": "Night"}
    return lookup.get(value.lower(), value)


# -----------------------------------------------------------------------------
# Excel data manager
# -----------------------------------------------------------------------------
class HospitalExcelDataManager:
    """Load and validate the supplied hospital HRM Excel workbook."""

    REQUIRED_SHEETS = {
        "staff",
        "staff_skills",
        "availability",
        "preferences",
        "demand_shift",
    }

    REQUIRED_COLUMNS = {
        "staff": {
            "staff_id",
            "contracted_hours_per_week",
            "home_unit",
            "role",
        },
        "staff_skills": {"staff_id", "skill_code"},
        "availability": {"staff_id", "date", "shift", "available"},
        "preferences": {
            "staff_id",
            "date",
            "shift",
            "preference_score",
            "request_off",
        },
        "demand_shift": {
            "date",
            "unit",
            "shift",
            "required_staff",
        },
    }

    def __init__(self, workbook_path: str):
        self.workbook_path = workbook_path
        if not os.path.exists(workbook_path):
            raise FileNotFoundError(f"Dataset not found: {workbook_path}")

    def load(self) -> Dict[str, pd.DataFrame]:
        logger.info("Loading hospital workbook: %s", self.workbook_path)
        xls = pd.ExcelFile(self.workbook_path)
        missing_sheets = self.REQUIRED_SHEETS - set(xls.sheet_names)
        if missing_sheets:
            raise ValueError(f"Workbook is missing required sheets: {sorted(missing_sheets)}")

        data = {
            name: pd.read_excel(self.workbook_path, sheet_name=name)
            for name in self.REQUIRED_SHEETS
        }

        for name, required in self.REQUIRED_COLUMNS.items():
            missing = required - set(data[name].columns)
            if missing:
                raise ValueError(f"Sheet '{name}' is missing columns: {sorted(missing)}")

        # Normalize dates and string keys without changing the source workbook.
        for name in ["availability", "preferences", "demand_shift"]:
            data[name]["date"] = pd.to_datetime(data[name]["date"])
            data[name]["shift"] = data[name]["shift"].map(normalize_shift)

        for name in ["staff", "staff_skills", "availability", "preferences"]:
            data[name]["staff_id"] = data[name]["staff_id"].astype(str)

        data["demand_shift"]["unit"] = data["demand_shift"]["unit"].astype(str)
        data["demand_shift"]["required_staff"] = (
            pd.to_numeric(data["demand_shift"]["required_staff"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        # Every min_* column is a skill requirement in the supplied dataset.
        self.skill_requirement_columns = [
            c for c in data["demand_shift"].columns if c.startswith("min_")
        ]
        if not self.skill_requirement_columns:
            raise ValueError("No min_* skill requirement columns found in demand_shift.")

        logger.info(
            "Loaded staff=%d, demand=%d, availability=%d, preferences=%d, skills=%d",
            len(data["staff"]),
            len(data["demand_shift"]),
            len(data["availability"]),
            len(data["preferences"]),
            len(data["staff_skills"]),
        )
        logger.info("Detected skill requirements: %s", self.skill_requirement_columns)
        return data


# -----------------------------------------------------------------------------
# Version 2 skill-aware CP-SAT scheduler
# -----------------------------------------------------------------------------
class SkillAwareIntelligentStaffScheduler:
    """
    CP-SAT baseline scheduler using the actual dataset structure.

    A demand row represents one unit/shift/day. x[staff, demand_row] is 1 when
    a staff member is assigned to that row.

    Total staffing and each min_* skill requirement are modeled separately.
    Skill shortage variables make the optimization robust to infeasible
    real-world scenarios without ever treating an unqualified staff member as
    qualified.
    """

    def __init__(self, config: Optional[V2SchedulingConfig] = None):
        self.config = config or V2SchedulingConfig()
        self.last_solution = None

    def _build_maps(
        self,
        staff_df: pd.DataFrame,
        availability_df: pd.DataFrame,
        preferences_df: pd.DataFrame,
        staff_skills_df: pd.DataFrame,
    ):
        availability_map: Dict[Tuple[str, str, str], int] = {}
        for r in availability_df.itertuples(index=False):
            availability_map[(str(r.staff_id), normalize_date(r.date), normalize_shift(r.shift))] = int(r.available)

        # The actual dataset has no unit in preferences, so preference is keyed
        # by staff/date/shift and applies to whichever unit is assigned.
        preference_map: Dict[Tuple[str, str, str], float] = {}
        request_off_map: Dict[Tuple[str, str, str], int] = {}
        for r in preferences_df.itertuples(index=False):
            key = (str(r.staff_id), normalize_date(r.date), normalize_shift(r.shift))
            preference_map[key] = float(r.preference_score) if pd.notna(r.preference_score) else 0.5
            request_off_map[key] = int(r.request_off) if pd.notna(r.request_off) else 0

        skill_map: Dict[str, set] = {}
        for r in staff_skills_df.itertuples(index=False):
            skill_map.setdefault(str(r.staff_id), set()).add(str(r.skill_code))

        return availability_map, preference_map, request_off_map, skill_map

    def _skill_name_from_column(self, column: str) -> str:
        return column[len("min_"):]

    def generate_roster(
        self,
        staff_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        availability_df: pd.DataFrame,
        preferences_df: pd.DataFrame,
        staff_skills_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Generate the Version 2 skill-aware baseline roster."""
        model = cp_model.CpModel()

        staff = staff_df.copy()
        demand = demand_df.copy()
        availability = availability_df.copy()
        preferences = preferences_df.copy()
        staff_skills = staff_skills_df.copy()

        staff["staff_id"] = staff["staff_id"].astype(str)
        demand["date"] = pd.to_datetime(demand["date"])
        demand["shift"] = demand["shift"].map(normalize_shift)
        availability["date"] = pd.to_datetime(availability["date"])
        availability["shift"] = availability["shift"].map(normalize_shift)
        preferences["date"] = pd.to_datetime(preferences["date"])
        preferences["shift"] = preferences["shift"].map(normalize_shift)
        staff_skills["staff_id"] = staff_skills["staff_id"].astype(str)

        availability_map, preference_map, request_off_map, skill_map = self._build_maps(
            staff, availability, preferences, staff_skills
        )

        staff_ids = staff["staff_id"].tolist()
        demand_rows = list(demand.itertuples(index=False))
        skill_columns = [c for c in demand.columns if c.startswith("min_")]

        # Decision variables.
        x: Dict[Tuple[str, int], cp_model.IntVar] = {}
        uncovered: Dict[int, cp_model.IntVar] = {}
        skill_shortage: Dict[Tuple[int, str], cp_model.IntVar] = {}

        for d_idx, row in enumerate(demand_rows):
            required = int(row.required_staff)
            uncovered[d_idx] = model.NewIntVar(0, required, f"uncovered_{d_idx}")
            for sid in staff_ids:
                x[(sid, d_idx)] = model.NewBoolVar(f"x_{sid}_{d_idx}")

            for col in skill_columns:
                required_skill = int(getattr(row, col) or 0)
                skill_shortage[(d_idx, col)] = model.NewIntVar(
                    0, max(0, required_skill), f"skill_shortage_{d_idx}_{col}"
                )

        # 1. Total staffing coverage.
        for d_idx, row in enumerate(demand_rows):
            assigned = [x[(sid, d_idx)] for sid in staff_ids]
            model.Add(sum(assigned) + uncovered[d_idx] >= int(row.required_staff))

        # 2. Availability and request-off constraints.
        for d_idx, row in enumerate(demand_rows):
            date_key = normalize_date(row.date)
            shift_key = normalize_shift(row.shift)
            for sid in staff_ids:
                if availability_map.get((sid, date_key, shift_key), 0) == 0:
                    model.Add(x[(sid, d_idx)] == 0)
                if request_off_map.get((sid, date_key, shift_key), 0) == 1:
                    model.Add(x[(sid, d_idx)] == 0)

        # 3. Skill-aware minimum coverage.
        #    Qualified staff only can satisfy the corresponding min_* demand.
        for d_idx, row in enumerate(demand_rows):
            for col in skill_columns:
                required_skill = int(getattr(row, col) or 0)
                if required_skill <= 0:
                    model.Add(skill_shortage[(d_idx, col)] == 0)
                    continue

                skill_code = self._skill_name_from_column(col)
                qualified_vars = [
                    x[(sid, d_idx)]
                    for sid in staff_ids
                    if skill_code in skill_map.get(sid, set())
                ]
                if qualified_vars:
                    model.Add(
                        sum(qualified_vars) + skill_shortage[(d_idx, col)] >= required_skill
                    )
                else:
                    model.Add(skill_shortage[(d_idx, col)] == required_skill)

        # 4. At most one shift per staff member per day.
        day_shift_indices: Dict[str, List[int]] = {}
        for d_idx, row in enumerate(demand_rows):
            day_shift_indices.setdefault(normalize_date(row.date), []).append(d_idx)

        for sid in staff_ids:
            for _, indices in day_shift_indices.items():
                model.Add(sum(x[(sid, i)] for i in indices) <= 1)

        # 5. Weekly contracted-hours constraints.
        staff_hours_map = {
            str(r.staff_id): int(r.contracted_hours_per_week)
            for r in staff.itertuples(index=False)
        }
        demand_meta = {
            i: {
                "date": pd.Timestamp(r.date),
                "shift": normalize_shift(r.shift),
                "week": pd.Timestamp(r.date).isocalendar().week,
                "year": pd.Timestamp(r.date).isocalendar().year,
            }
            for i, r in enumerate(demand_rows)
        }

        week_indices: Dict[Tuple[int, int], List[int]] = {}
        for d_idx, meta in demand_meta.items():
            week_indices.setdefault((meta["year"], meta["week"]), []).append(d_idx)

        for sid in staff_ids:
            weekly_hours = staff_hours_map.get(sid, self.config.max_hours_default)
            for week_key, indices in week_indices.items():
                terms = [
                    x[(sid, i)] * self.config.shift_hours.get(demand_meta[i]["shift"], 8)
                    for i in indices
                ]
                model.Add(sum(terms) <= weekly_hours)

        # 6. Configured forbidden transitions, e.g. Night -> Day next day.
        for sid in staff_ids:
            for shift_a, shift_b in self.config.forbidden_pairs:
                shift_a = normalize_shift(shift_a)
                shift_b = normalize_shift(shift_b)
                for i, meta_a in demand_meta.items():
                    if meta_a["shift"] != shift_a:
                        continue
                    next_date = meta_a["date"] + pd.Timedelta(days=1)
                    for j, meta_b in demand_meta.items():
                        if meta_b["date"] == next_date and meta_b["shift"] == shift_b:
                            model.Add(x[(sid, i)] + x[(sid, j)] <= 1)

        # 7. Workload fairness: minimize deviation from an even assignment count
        #    over the complete planning horizon. This is a soft objective only.
        max_assignments = len(demand_rows)
        assignment_count: Dict[str, cp_model.IntVar] = {}
        fairness_deviation: List[cp_model.IntVar] = []
        target_assignments = max(0, round(sum(int(r.required_staff) for r in demand_rows) / len(staff_ids)))

        for sid in staff_ids:
            count = model.NewIntVar(0, max_assignments, f"assignment_count_{sid}")
            model.Add(count == sum(x[(sid, i)] for i in range(len(demand_rows))))
            assignment_count[sid] = count
            dev = model.NewIntVar(0, max_assignments, f"fairness_dev_{sid}")
            model.AddAbsEquality(dev, count - target_assignments)
            fairness_deviation.append(dev)

        # 8. Preference dissatisfaction.
        preference_terms = []
        for d_idx, row in enumerate(demand_rows):
            date_key = normalize_date(row.date)
            shift_key = normalize_shift(row.shift)
            for sid in staff_ids:
                pref = preference_map.get((sid, date_key, shift_key), 0.5)
                penalty = int(round(max(0.0, min(1.0, 1.0 - pref)) * 100))
                if penalty:
                    preference_terms.append(x[(sid, d_idx)] * penalty)

        # Objective: hard constraints stay hard; slacks are minimized first via
        # very large coefficients, followed by preferences/fairness.
        objective = []
        objective.extend(self.config.uncovered_weight * uncovered[i] for i in uncovered)
        objective.extend(
            self.config.skill_shortage_weight * skill_shortage[key]
            for key in skill_shortage
        )
        objective.extend(self.config.preference_weight * term for term in preference_terms)
        objective.extend(self.config.fairness_weight * dev for dev in fairness_deviation)
        model.Minimize(sum(objective))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(self.config.max_solver_seconds)
        solver.parameters.num_search_workers = int(self.config.num_search_workers)

        logger.info(
            "Solving V2 baseline: %d staff, %d demand rows, %d skill columns",
            len(staff_ids), len(demand_rows), len(skill_columns),
        )
        start = time.perf_counter()
        status = solver.Solve(model)
        elapsed = time.perf_counter() - start

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"No feasible V2 roster found. CP-SAT status: {solver.StatusName(status)}"
            )

        # ------------------------------------------------------------------
        # Build roster and detailed metrics.
        # ------------------------------------------------------------------
        output_rows = []
        workload_hours = {sid: 0 for sid in staff_ids}
        weekly_workload: Dict[Tuple[str, int, int], float] = {}
        total_uncovered = 0
        total_required = 0
        total_assigned = 0
        preference_count = 0
        satisfied_preferences = 0
        request_off_violations = 0

        demand_metric_rows = []
        skill_metric_rows = []

        for d_idx, row in enumerate(demand_rows):
            date_key = normalize_date(row.date)
            shift_key = normalize_shift(row.shift)
            assigned_staff = []

            for sid in staff_ids:
                if solver.Value(x[(sid, d_idx)]) == 1:
                    pref = preference_map.get((sid, date_key, shift_key), 0.5)
                    hours = self.config.shift_hours.get(shift_key, 8)
                    workload_hours[sid] += hours

                    ts = pd.Timestamp(row.date)
                    iso = ts.isocalendar()
                    week_key = (sid, int(iso.year), int(iso.week))
                    weekly_workload[week_key] = weekly_workload.get(week_key, 0) + hours

                    preference_count += 1
                    if pref >= 0.6:
                        satisfied_preferences += 1
                    if request_off_map.get((sid, date_key, shift_key), 0) == 1:
                        request_off_violations += 1

                    assigned_staff.append(sid)
                    output_rows.append(
                        {
                            "staff_id": sid,
                            "date": date_key,
                            "unit": str(row.unit),
                            "shift": shift_key,
                            "assigned": 1,
                            "preference_score": pref,
                            "shift_hours": hours,
                        }
                    )

            required = int(row.required_staff)
            assigned = len(assigned_staff)
            uncovered_value = max(0, required - assigned)
            total_required += required
            total_assigned += assigned
            total_uncovered += uncovered_value

            demand_metric_rows.append(
                {
                    "date": date_key,
                    "unit": str(row.unit),
                    "shift": shift_key,
                    "required_staff": required,
                    "assigned_staff": assigned,
                    "uncovered_staff": uncovered_value,
                }
            )

            for col in skill_columns:
                required_skill = int(getattr(row, col) or 0)
                if required_skill <= 0:
                    continue
                skill_code = self._skill_name_from_column(col)
                qualified_assigned = sum(
                    1 for sid in assigned_staff if skill_code in skill_map.get(sid, set())
                )
                shortfall = max(0, required_skill - qualified_assigned)
                skill_metric_rows.append(
                    {
                        "date": date_key,
                        "unit": str(row.unit),
                        "shift": shift_key,
                        "skill_code": skill_code,
                        "required_skill": required_skill,
                        "qualified_assigned": qualified_assigned,
                        "skill_shortfall": shortfall,
                    }
                )

        roster_df = pd.DataFrame(output_rows)
        demand_metrics_df = pd.DataFrame(demand_metric_rows)
        skill_metrics_df = pd.DataFrame(skill_metric_rows)

        contracted_hours = {
            str(r.staff_id): int(r.contracted_hours_per_week)
            for r in staff.itertuples(index=False)
        }
        weekly_overage = 0.0
        for (sid, _, _), hours in weekly_workload.items():
            weekly_overage += max(0.0, hours - contracted_hours.get(sid, self.config.max_hours_default))

        skill_required_total = float(skill_metrics_df["required_skill"].sum()) if not skill_metrics_df.empty else 0.0
        skill_shortfall_total = float(skill_metrics_df["skill_shortfall"].sum()) if not skill_metrics_df.empty else 0.0

        metrics = {
            "solver_status": solver.StatusName(status),
            "solve_time_seconds": float(elapsed),
            "planning_days": int(demand["date"].nunique()),
            "demand_rows": int(len(demand_rows)),
            "staff_count": int(len(staff_ids)),
            "total_required_positions": float(total_required),
            "total_assigned_positions": float(total_assigned),
            "uncovered_positions": float(total_uncovered),
            "staffing_coverage_percent": safe_divide(total_assigned, total_required) * 100.0,
            "skill_requirements_total": skill_required_total,
            "skill_shortfall_total": skill_shortfall_total,
            "skill_fulfillment_percent": (
                safe_divide(skill_required_total - skill_shortfall_total, skill_required_total) * 100.0
                if skill_required_total else 100.0
            ),
            "preference_satisfaction_percent": safe_divide(satisfied_preferences, preference_count) * 100.0,
            "request_off_violations": float(request_off_violations),
            "weekly_hours_overage": float(weekly_overage),
            "workload_gini": compute_gini(workload_hours.values()),
            "mean_workload_hours": float(np.mean(list(workload_hours.values()))),
            "std_workload_hours": float(np.std(list(workload_hours.values()))),
            "max_workload_hours": float(max(workload_hours.values()) if workload_hours else 0),
        }

        self.last_solution = {
            "roster": roster_df,
            "demand_metrics": demand_metrics_df,
            "skill_metrics": skill_metrics_df,
            "metrics": metrics,
            "workload_hours": workload_hours,
            "weekly_workload": weekly_workload,
            "skill_map": skill_map,
        }
        return roster_df, metrics

    def validate_roster(
        self,
        roster_df: pd.DataFrame,
        staff_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        availability_df: pd.DataFrame,
        preferences_df: pd.DataFrame,
        staff_skills_df: pd.DataFrame,
    ) -> Dict[str, object]:
        """Independent post-solve validation for research reporting."""
        if roster_df.empty:
            return {
                "valid": False,
                "errors": ["Roster is empty."],
                "checks": {},
            }

        errors: List[str] = []
        checks: Dict[str, bool] = {}

        roster = roster_df.copy()
        roster["date"] = pd.to_datetime(roster["date"])
        roster["shift"] = roster["shift"].map(normalize_shift)

        # Duplicate staff/date check.
        duplicate_count = int(roster.duplicated(["staff_id", "date"]).sum())
        checks["one_shift_per_day"] = duplicate_count == 0
        if duplicate_count:
            errors.append(f"Found {duplicate_count} duplicate staff/date assignments.")

        # Availability check.
        av = availability_df.copy()
        av["date"] = pd.to_datetime(av["date"])
        av["shift"] = av["shift"].map(normalize_shift)
        av_keys = set(
            zip(av["staff_id"].astype(str), av["date"].map(normalize_date), av["shift"], av["available"].astype(int))
        )
        unavailable_assignments = 0
        for r in roster.itertuples(index=False):
            if (str(r.staff_id), normalize_date(r.date), normalize_shift(r.shift), 1) not in av_keys:
                unavailable_assignments += 1
        checks["availability"] = unavailable_assignments == 0
        if unavailable_assignments:
            errors.append(f"Found {unavailable_assignments} assignments without availability=1.")

        # Request-off check.
        pref = preferences_df.copy()
        pref["date"] = pd.to_datetime(pref["date"])
        pref["shift"] = pref["shift"].map(normalize_shift)
        off_keys = set(
            zip(
                pref.loc[pref["request_off"] == 1, "staff_id"].astype(str),
                pref.loc[pref["request_off"] == 1, "date"].map(normalize_date),
                pref.loc[pref["request_off"] == 1, "shift"],
            )
        )
        off_violations = sum(
            (str(r.staff_id), normalize_date(r.date), normalize_shift(r.shift)) in off_keys
            for r in roster.itertuples(index=False)
        )
        checks["request_off"] = off_violations == 0
        if off_violations:
            errors.append(f"Found {off_violations} request-off violations.")

        # Weekly hours check.
        staff_hours = dict(
            zip(
                staff_df["staff_id"].astype(str),
                staff_df["contracted_hours_per_week"].astype(int),
            )
        )
        roster["iso_year"] = roster["date"].dt.isocalendar().year.astype(int)
        roster["iso_week"] = roster["date"].dt.isocalendar().week.astype(int)
        roster["hours"] = roster["shift"].map(self.config.shift_hours).fillna(8)
        weekly = roster.groupby(["staff_id", "iso_year", "iso_week"])["hours"].sum().reset_index()
        weekly["limit"] = weekly["staff_id"].map(staff_hours).fillna(self.config.max_hours_default)
        hour_violations = int((weekly["hours"] > weekly["limit"]).sum())
        checks["weekly_hours"] = hour_violations == 0
        if hour_violations:
            errors.append(f"Found {hour_violations} staff-weeks over contracted hours.")

        # Skill validation.
        skill_map: Dict[str, set] = {}
        for r in staff_skills_df.itertuples(index=False):
            skill_map.setdefault(str(r.staff_id), set()).add(str(r.skill_code))

        skill_cols = [c for c in demand_df.columns if c.startswith("min_")]
        demand_lookup = {
            (normalize_date(r.date), str(r.unit), normalize_shift(r.shift)): r
            for r in demand_df.itertuples(index=False)
        }
        skill_shortfall = 0
        total_skill_requirement = 0
        for key, group in roster.groupby([roster["date"].map(normalize_date), "unit", "shift"]):
            drow = demand_lookup.get(key)
            if drow is None:
                continue
            staff_in_group = group["staff_id"].astype(str).tolist()
            for col in skill_cols:
                required = int(getattr(drow, col) or 0)
                if required <= 0:
                    continue
                code = self._skill_name_from_column(col)
                qualified = sum(code in skill_map.get(sid, set()) for sid in staff_in_group)
                skill_shortfall += max(0, required - qualified)
                total_skill_requirement += required

        checks["skill_requirements"] = skill_shortfall == 0

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "checks": checks,
            "duplicate_staff_day": duplicate_count,
            "unavailable_assignments": unavailable_assignments,
            "request_off_violations": off_violations,
            "weekly_hour_violations": hour_violations,
            "skill_shortfall": skill_shortfall,
            "skill_requirement_total": total_skill_requirement,
        }


# -----------------------------------------------------------------------------
# Version 2 runner/exporter
# -----------------------------------------------------------------------------
def run_v2_baseline(
    dataset_path: str,
    output_dir: str = "v2_outputs",
    config: Optional[V2SchedulingConfig] = None,
) -> Dict[str, str]:
    """Load the actual workbook, solve the V2 baseline, validate and export."""
    os.makedirs(output_dir, exist_ok=True)

    manager = HospitalExcelDataManager(dataset_path)
    data = manager.load()
    scheduler = SkillAwareIntelligentStaffScheduler(config)

    roster_df, metrics = scheduler.generate_roster(
        staff_df=data["staff"],
        demand_df=data["demand_shift"],
        availability_df=data["availability"],
        preferences_df=data["preferences"],
        staff_skills_df=data["staff_skills"],
    )

    validation = scheduler.validate_roster(
        roster_df=roster_df,
        staff_df=data["staff"],
        demand_df=data["demand_shift"],
        availability_df=data["availability"],
        preferences_df=data["preferences"],
        staff_skills_df=data["staff_skills"],
    )

    roster_path = os.path.join(output_dir, "v2_skill_aware_baseline_roster.csv")
    demand_path = os.path.join(output_dir, "v2_demand_coverage.csv")
    skill_path = os.path.join(output_dir, "v2_skill_coverage.csv")
    metrics_path = os.path.join(output_dir, "v2_baseline_metrics.json")
    validation_path = os.path.join(output_dir, "v2_validation_report.json")

    roster_df.to_csv(roster_path, index=False)
    scheduler.last_solution["demand_metrics"].to_csv(demand_path, index=False)
    scheduler.last_solution["skill_metrics"].to_csv(skill_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    logger.info("V2 baseline completed.")
    logger.info("Metrics: %s", metrics)
    logger.info("Validation: %s", validation)

    return {
        "roster": roster_path,
        "demand_coverage": demand_path,
        "skill_coverage": skill_path,
        "metrics": metrics_path,
        "validation": validation_path,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Version 2 hospital scheduling baseline.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to hospital_hrm_dataset.xlsx",
    )
    parser.add_argument(
        "--output",
        default="v2_outputs",
        help="Directory for Version 2 outputs",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=120,
        help="CP-SAT time limit in seconds",
    )
    args = parser.parse_args()

    cfg = V2SchedulingConfig(max_solver_seconds=args.time_limit)
    outputs = run_v2_baseline(args.dataset, args.output, cfg)
    print("\nVersion 2 outputs:")
    for key, path in outputs.items():
        print(f"- {key}: {path}")
