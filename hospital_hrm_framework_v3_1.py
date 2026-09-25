"""
Hospital HRM Framework - Version 3.1
==================================

Version 3.1 builds directly on Version 3 and adds transition diagnostics.

Purpose
-------
Turn the validated, skill-aware CP-SAT roster from Version 2 into a dynamic
hospital workforce environment suitable for reinforcement learning.

Key additions over V2
---------------------
1. Starts from a V2 baseline roster instead of inventing a separate schedule.
2. Applies controlled dynamic events:
   - staff absence
   - demand surge / shortage increase
3. Reallocates REAL staff members between units on the same date/shift.
4. Generates only operationally valid candidate moves:
   - staff must be available
   - no request-off violation
   - staff cannot already be assigned elsewhere on that date/shift
   - target unit can require a skill the staff member possesses
   - weekly contracted hours are respected
   - basic rest rules are respected
5. State includes total and skill-specific shortage information.
6. Action is a candidate staff-level move, not an abstract "move N staff" action.
7. Reward is based on improvement in coverage/skill coverage while accounting
   for fairness, overtime, preferences and unnecessary changes.
8. Includes a deterministic scenario runner and smoke test.
9. Adds source/target transition diagnostics and reward-component diagnostics.
10. Records the event associated with the action date rather than the next date.

This is a research prototype and decision-support environment, not a clinical
production scheduling system.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from hospital_hrm_framework_v2 import (
    HospitalExcelDataManager,
    SkillAwareIntelligentStaffScheduler,
    V2SchedulingConfig,
    compute_gini,
    normalize_date,
    normalize_shift,
    safe_divide,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("hospital_hrm_v3")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class V3EnvironmentConfig:
    seed: int = 42
    episode_days: int = 7
    surge_probability: float = 0.35
    surge_staff_min: int = 1
    surge_staff_max: int = 4
    absence_probability: float = 0.12
    max_absences_per_day: int = 3

    # Candidate/reward settings.
    max_candidates: int = 64
    no_op_action: int = 0
    coverage_reward: float = 10.0
    skill_reward: float = 12.0
    shortage_penalty: float = 10.0
    skill_shortage_penalty: float = 12.0
    fairness_penalty: float = 2.0
    overtime_penalty: float = 3.0
    preference_penalty: float = 1.0
    change_penalty: float = 0.25
    invalid_action_penalty: float = 4.0

    shift_hours: Dict[str, int] = field(
        default_factory=lambda: {"Day": 8, "Evening": 8, "Night": 8}
    )
    forbidden_pairs: List[Tuple[str, str]] = field(
        default_factory=lambda: [("Night", "Day")]
    )


# -----------------------------------------------------------------------------
# Dynamic environment
# -----------------------------------------------------------------------------
class DynamicHospitalWorkforceEnvironment:
    """A staff-level dynamic scheduling environment built on the V2 roster."""

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        baseline_roster: pd.DataFrame,
        config: Optional[V3EnvironmentConfig] = None,
    ):
        self.data = data
        self.config = config or V3EnvironmentConfig()
        self.rng = random.Random(self.config.seed)

        self.staff = data["staff"].copy()
        self.staff["staff_id"] = self.staff["staff_id"].astype(str)
        self.staff["home_unit"] = self.staff["home_unit"].astype(str)
        self.staff_hours = dict(
            zip(
                self.staff["staff_id"],
                pd.to_numeric(self.staff["contracted_hours_per_week"], errors="coerce")
                .fillna(40)
                .astype(float),
            )
        )

        self.availability = data["availability"].copy()
        self.availability["staff_id"] = self.availability["staff_id"].astype(str)
        self.availability["date"] = pd.to_datetime(self.availability["date"])
        self.availability["shift"] = self.availability["shift"].map(normalize_shift)

        self.preferences = data["preferences"].copy()
        self.preferences["staff_id"] = self.preferences["staff_id"].astype(str)
        self.preferences["date"] = pd.to_datetime(self.preferences["date"])
        self.preferences["shift"] = self.preferences["shift"].map(normalize_shift)

        self.staff_skills = data["staff_skills"].copy()
        self.staff_skills["staff_id"] = self.staff_skills["staff_id"].astype(str)
        self.staff_skills["skill_code"] = self.staff_skills["skill_code"].astype(str)
        self.skill_map: Dict[str, set] = {}
        for row in self.staff_skills.itertuples(index=False):
            self.skill_map.setdefault(str(row.staff_id), set()).add(str(row.skill_code))

        self.demand = data["demand_shift"].copy()
        self.demand["date"] = pd.to_datetime(self.demand["date"])
        self.demand["shift"] = self.demand["shift"].map(normalize_shift)
        self.demand["unit"] = self.demand["unit"].astype(str)
        self.skill_columns = [c for c in self.demand.columns if c.startswith("min_")]

        self.baseline = self._normalize_roster(baseline_roster)
        self.roster = self.baseline.copy()
        self.current_demand = self.demand.copy()
        self.absent_staff: set[str] = set()
        self.current_date: Optional[pd.Timestamp] = None
        self.current_shift: Optional[str] = None
        self.candidates: List[Dict[str, object]] = []
        self.last_event: Dict[str, object] = {}
        self.last_action: Dict[str, object] = {"type": "no_op"}
        self.done = False

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_roster(roster: pd.DataFrame) -> pd.DataFrame:
        r = roster.copy()
        if r.empty:
            raise ValueError("Baseline roster is empty.")
        required = {"staff_id", "date", "unit", "shift"}
        missing = required - set(r.columns)
        if missing:
            raise ValueError(f"Baseline roster missing columns: {sorted(missing)}")
        r["staff_id"] = r["staff_id"].astype(str)
        r["date"] = pd.to_datetime(r["date"])
        r["unit"] = r["unit"].astype(str)
        r["shift"] = r["shift"].map(normalize_shift)
        if "shift_hours" not in r.columns:
            r["shift_hours"] = 8
        r["shift_hours"] = pd.to_numeric(r["shift_hours"], errors="coerce").fillna(8)
        if "preference_score" not in r.columns:
            r["preference_score"] = 0.5
        r["preference_score"] = pd.to_numeric(r["preference_score"], errors="coerce").fillna(0.5)
        return r

    def _availability(self, staff_id: str, date: pd.Timestamp, shift: str) -> bool:
        rows = self.availability[
            (self.availability["staff_id"] == str(staff_id))
            & (self.availability["date"] == pd.Timestamp(date))
            & (self.availability["shift"] == normalize_shift(shift))
        ]
        if rows.empty:
            return False
        return bool(int(rows.iloc[0]["available"]) == 1)

    def _request_off(self, staff_id: str, date: pd.Timestamp, shift: str) -> bool:
        rows = self.preferences[
            (self.preferences["staff_id"] == str(staff_id))
            & (self.preferences["date"] == pd.Timestamp(date))
            & (self.preferences["shift"] == normalize_shift(shift))
        ]
        return bool((rows["request_off"].fillna(0).astype(int) == 1).any())

    def _preference(self, staff_id: str, date: pd.Timestamp, shift: str) -> float:
        rows = self.preferences[
            (self.preferences["staff_id"] == str(staff_id))
            & (self.preferences["date"] == pd.Timestamp(date))
            & (self.preferences["shift"] == normalize_shift(shift))
        ]
        if rows.empty:
            return 0.5
        return float(pd.to_numeric(rows.iloc[0]["preference_score"], errors="coerce") or 0.5)

    def _week_key(self, date: pd.Timestamp) -> Tuple[int, int]:
        iso = pd.Timestamp(date).isocalendar()
        return int(iso.year), int(iso.week)

    def _weekly_hours(self, staff_id: str, date: pd.Timestamp, roster: Optional[pd.DataFrame] = None) -> float:
        r = self.roster if roster is None else roster
        if r.empty:
            return 0.0
        y, w = self._week_key(date)
        dates = pd.to_datetime(r["date"])
        iso = dates.dt.isocalendar()
        mask = (
            (r["staff_id"].astype(str) == str(staff_id))
            & (iso.year.astype(int) == y)
            & (iso.week.astype(int) == w)
        )
        return float(r.loc[mask, "shift_hours"].sum())

    def _has_other_assignment_same_day(
        self, staff_id: str, date: pd.Timestamp, shift: str, roster: Optional[pd.DataFrame] = None
    ) -> bool:
        r = self.roster if roster is None else roster
        mask = (
            (r["staff_id"].astype(str) == str(staff_id))
            & (r["date"] == pd.Timestamp(date))
        )
        if not mask.any():
            return False
        return not bool((r.loc[mask, "shift"] == normalize_shift(shift)).all())

    def _rest_violation(self, staff_id: str, date: pd.Timestamp, target_shift: str, roster: Optional[pd.DataFrame] = None) -> bool:
        r = self.roster if roster is None else roster
        sid = str(staff_id)
        date = pd.Timestamp(date)
        prior = r[(r["staff_id"].astype(str) == sid) & (r["date"] == date - pd.Timedelta(days=1))]
        if prior.empty:
            return False
        prior_shifts = set(prior["shift"].map(normalize_shift))
        return any((p, normalize_shift(target_shift)) in self.config.forbidden_pairs for p in prior_shifts)

    # ------------------------------------------------------------------
    # Scenario creation
    # ------------------------------------------------------------------
    def _scenario_dates(self, start_date: Optional[str] = None) -> List[pd.Timestamp]:
        available_dates = sorted(pd.to_datetime(self.demand["date"].unique()))
        if start_date is not None:
            start = pd.Timestamp(start_date)
            available_dates = [d for d in available_dates if d >= start]
        return [pd.Timestamp(d) for d in available_dates[: self.config.episode_days]]

    def reset(self, start_date: Optional[str] = None, forced_events: Optional[List[Dict[str, object]]] = None):
        dates = self._scenario_dates(start_date)
        if not dates:
            raise ValueError("No demand dates available for the requested episode.")
        self.episode_dates = dates
        self.current_index = 0
        self.current_date = dates[0]
        self.current_shift = "Day"
        self.roster = self.baseline.copy()
        self.current_demand = self.demand.copy()
        self.absent_staff = set()
        self.candidates = []
        self.last_action = {"type": "no_op"}
        self.done = False

        self.forced_events = forced_events or []
        self.last_event = self._inject_event_for_current_date()
        self._remove_absent_assignments()
        self.candidates = self.get_valid_actions()
        return self.get_state()

    def _inject_event_for_current_date(self) -> Dict[str, object]:
        date = pd.Timestamp(self.current_date)
        forced = None
        for event in getattr(self, "forced_events", []):
            if pd.Timestamp(event.get("date")) == date:
                forced = event
                break

        event: Dict[str, object] = {
            "date": normalize_date(date),
            "demand_surge": [],
            "absent_staff": [],
        }

        if forced:
            surge = forced.get("demand_surge", [])
            absences = [str(x) for x in forced.get("absent_staff", [])]
        else:
            surge = []
            if self.rng.random() < self.config.surge_probability:
                day_rows = self.demand[self.demand["date"] == date]
                if not day_rows.empty:
                    sample_n = min(2, len(day_rows))
                    for idx in self.rng.sample(list(day_rows.index), sample_n):
                        surge.append({
                            "unit": str(day_rows.loc[idx, "unit"]),
                            "shift": str(day_rows.loc[idx, "shift"]),
                            "extra_staff": self.rng.randint(
                                self.config.surge_staff_min, self.config.surge_staff_max
                            ),
                        })
            absences = []
            day_staff = self.roster[self.roster["date"] == date]["staff_id"].astype(str).unique().tolist()
            self.rng.shuffle(day_staff)
            for sid in day_staff:
                if len(absences) >= self.config.max_absences_per_day:
                    break
                if self.rng.random() < self.config.absence_probability:
                    absences.append(sid)

        self.absent_staff = set(absences)
        event["absent_staff"] = sorted(self.absent_staff)

        for item in surge:
            unit = str(item["unit"])
            shift = normalize_shift(str(item["shift"]))
            extra = int(item.get("extra_staff", 0))
            mask = (
                (self.current_demand["date"] == date)
                & (self.current_demand["unit"] == unit)
                & (self.current_demand["shift"] == shift)
            )
            if mask.any():
                self.current_demand.loc[mask, "required_staff"] = (
                    self.current_demand.loc[mask, "required_staff"].astype(int) + extra
                )
            else:
                logger.warning("Forced surge target not found: %s/%s/%s", normalize_date(date), unit, shift)
            event["demand_surge"].append({"unit": unit, "shift": shift, "extra_staff": extra})

        return event

    def _remove_absent_assignments(self):
        if not self.absent_staff:
            return
        self.roster = self.roster[
            ~self.roster["staff_id"].astype(str).isin(self.absent_staff)
            | (self.roster["date"] != pd.Timestamp(self.current_date))
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Metrics and state
    # ------------------------------------------------------------------
    def _demand_row(self, date: pd.Timestamp, unit: str, shift: str):
        rows = self.current_demand[
            (self.current_demand["date"] == pd.Timestamp(date))
            & (self.current_demand["unit"] == str(unit))
            & (self.current_demand["shift"] == normalize_shift(shift))
        ]
        return rows.iloc[0] if not rows.empty else None

    def _group_metrics(self, date: pd.Timestamp, unit: str, shift: str) -> Dict[str, object]:
        demand = self._demand_row(date, unit, shift)
        if demand is None:
            return {"required": 0, "assigned": 0, "shortage": 0, "skill_shortage": 0, "skill_details": {}}
        group = self.roster[
            (self.roster["date"] == pd.Timestamp(date))
            & (self.roster["unit"] == str(unit))
            & (self.roster["shift"] == normalize_shift(shift))
        ]
        assigned_ids = group["staff_id"].astype(str).tolist()
        required = int(demand["required_staff"])
        assigned = len(assigned_ids)
        skill_details = {}
        skill_shortage = 0
        for col in self.skill_columns:
            req = int(demand[col] or 0)
            if req <= 0:
                continue
            code = col[len("min_"):]
            qualified = sum(code in self.skill_map.get(sid, set()) for sid in assigned_ids)
            short = max(0, req - qualified)
            skill_shortage += short
            skill_details[code] = {"required": req, "qualified": qualified, "shortage": short}
        return {
            "required": required,
            "assigned": assigned,
            "shortage": max(0, required - assigned),
            "skill_shortage": skill_shortage,
            "skill_details": skill_details,
        }

    def _all_metrics(self) -> Dict[str, float]:
        date = pd.Timestamp(self.current_date)
        day_demand = self.current_demand[self.current_demand["date"] == date]
        total_required = 0
        total_assigned = 0
        total_shortage = 0
        total_skill_required = 0
        total_skill_shortage = 0

        for row in day_demand.itertuples(index=False):
            m = self._group_metrics(date, row.unit, row.shift)
            total_required += m["required"]
            total_assigned += m["assigned"]
            total_shortage += m["shortage"]
            for detail in m["skill_details"].values():
                total_skill_required += detail["required"]
                total_skill_shortage += detail["shortage"]

        # Workload/fairness for staff scheduled in the current ISO week.
        y, w = self._week_key(date)
        dates = pd.to_datetime(self.roster["date"])
        iso = dates.dt.isocalendar()
        week_roster = self.roster[
            (iso.year.astype(int) == y) & (iso.week.astype(int) == w)
        ]
        workloads = (
            week_roster.groupby("staff_id")["shift_hours"].sum().reindex(self.staff_hours.keys(), fill_value=0)
        )
        workload_gini = compute_gini(workloads.values)
        overtime = float(
            sum(max(0.0, float(hours) - self.staff_hours.get(str(sid), 40.0)) for sid, hours in workloads.items())
        )

        pref_values = []
        for row in self.roster[self.roster["date"] == date].itertuples(index=False):
            pref_values.append(self._preference(str(row.staff_id), date, str(row.shift)))
        preference_score = float(np.mean(pref_values)) if pref_values else 0.5

        return {
            "required": float(total_required),
            "assigned": float(total_assigned),
            "shortage": float(total_shortage),
            "coverage": safe_divide(total_assigned, total_required),
            "skill_required": float(total_skill_required),
            "skill_shortage": float(total_skill_shortage),
            "skill_coverage": safe_divide(total_skill_required - total_skill_shortage, total_skill_required),
            "workload_gini": float(workload_gini),
            "overtime": overtime,
            "preference_score": preference_score,
        }

    def _state_components(self) -> Dict[str, float]:
        date = pd.Timestamp(self.current_date)
        components: Dict[str, float] = {}
        units = sorted(self.current_demand[self.current_demand["date"] == date]["unit"].unique())
        shifts = ["Day", "Evening", "Night"]
        for unit in units:
            for shift in shifts:
                m = self._group_metrics(date, unit, shift)
                components[f"{unit}_{shift}_coverage"] = float(safe_divide(m["assigned"], m["required"]))
                components[f"{unit}_{shift}_shortage"] = float(m["shortage"])
                components[f"{unit}_{shift}_skill_shortage"] = float(m["skill_shortage"])
        return components

    def get_state(self) -> np.ndarray:
        """Return a fixed-length numeric state for the current episode step."""
        metrics = self._all_metrics()
        comp = self._state_components()
        base = [
            metrics["coverage"],
            metrics["shortage"],
            metrics["skill_coverage"],
            metrics["skill_shortage"],
            metrics["workload_gini"],
            metrics["overtime"],
            metrics["preference_score"],
            len(self.absent_staff) / max(1, len(self.staff)),
            len(self.last_event.get("demand_surge", [])) / max(1, len(self.current_demand[self.current_demand["date"] == self.current_date])),
            self.current_index / max(1, len(self.episode_dates) - 1),
        ]
        # Fixed ordering is determined by sorted component keys.
        return np.asarray(base + [comp[k] for k in sorted(comp)], dtype=np.float32)

    # ------------------------------------------------------------------
    # Candidate actions
    # ------------------------------------------------------------------
    def _target_units(self, date: pd.Timestamp) -> List[Tuple[str, str]]:
        rows = self.current_demand[self.current_demand["date"] == date]
        scored = []
        for row in rows.itertuples(index=False):
            m = self._group_metrics(date, row.unit, row.shift)
            # Prioritize shortage, then skill shortage.
            score = 1000 * m["shortage"] + 100 * m["skill_shortage"]
            scored.append((score, str(row.unit), normalize_shift(row.shift)))
        scored.sort(reverse=True)
        return [(u, s) for _, u, s in scored if _ > 0]

    def _source_assignments(self, date: pd.Timestamp) -> pd.DataFrame:
        return self.roster[self.roster["date"] == date].copy()

    def _is_valid_move(self, sid: str, source_unit: str, source_shift: str, target_unit: str, target_shift: str) -> bool:
        date = pd.Timestamp(self.current_date)
        if source_shift != target_shift:
            return False  # V3 keeps reallocation within the same shift.
        if sid in self.absent_staff:
            return False
        if not self._availability(sid, date, target_shift):
            return False
        if self._request_off(sid, date, target_shift):
            return False
        if self._rest_violation(sid, date, target_shift):
            return False
        # A same-day unit reallocation does not add another shift; it only
        # changes the unit attached to the staff member's existing shift.
        # Therefore the weekly-hour total remains unchanged.
        current_week = self._weekly_hours(sid, date)
        if current_week > self.staff_hours.get(sid, 40.0):
            return False

        # Staff must possess at least one skill that is currently relevant to
        # the target unit/shift when the target has skill requirements. For a
        # target with no skill shortage, ordinary eligible staff are allowed.
        target_metrics = self._group_metrics(date, target_unit, target_shift)
        target_row = self._demand_row(date, target_unit, target_shift)
        if target_row is not None:
            required_skills = []
            for col in self.skill_columns:
                req = int(target_row[col] or 0)
                if req > 0:
                    code = col[len("min_"):]
                    qualified = target_metrics["skill_details"].get(code, {}).get("qualified", 0)
                    required = target_metrics["skill_details"].get(code, {}).get("required", 0)
                    if qualified < required:
                        required_skills.append(code)
            if required_skills and not any(code in self.skill_map.get(sid, set()) for code in required_skills):
                return False
        return True

    def get_valid_actions(self) -> List[Dict[str, object]]:
        """Return [no-op, candidate staff-level moves]."""
        date = pd.Timestamp(self.current_date)
        actions: List[Dict[str, object]] = [{"type": "no_op"}]
        targets = self._target_units(date)
        assignments = self._source_assignments(date)

        # Source units with less urgent shortage are considered first.
        source_rows = []
        for row in assignments.itertuples(index=False):
            m = self._group_metrics(date, row.unit, row.shift)
            source_rows.append((m["shortage"] * 100 + m["skill_shortage"], row))
        source_rows.sort(key=lambda x: x[0])

        for target_unit, target_shift in targets:
            target_m = self._group_metrics(date, target_unit, target_shift)
            for _, row in source_rows:
                if str(row.unit) == target_unit and normalize_shift(row.shift) == target_shift:
                    continue
                # Do not strip a unit below its required total staffing unless
                # the target shortage is more severe than the source situation.
                source_m = self._group_metrics(date, str(row.unit), normalize_shift(row.shift))
                if source_m["assigned"] <= source_m["required"] and target_m["shortage"] <= source_m["shortage"]:
                    continue
                sid = str(row.staff_id)
                if not self._is_valid_move(
                    sid, str(row.unit), normalize_shift(row.shift), target_unit, target_shift
                ):
                    continue
                action = {
                    "type": "reassign",
                    "staff_id": sid,
                    "date": normalize_date(date),
                    "source_unit": str(row.unit),
                    "source_shift": normalize_shift(row.shift),
                    "target_unit": target_unit,
                    "target_shift": target_shift,
                }
                actions.append(action)
                if len(actions) >= self.config.max_candidates + 1:
                    return actions
        return actions

    # ------------------------------------------------------------------
    # Environment transition and reward
    # ------------------------------------------------------------------
    def _apply_action(self, action: Dict[str, object]) -> bool:
        if action.get("type") == "no_op":
            return True
        if action.get("type") != "reassign":
            return False

        sid = str(action["staff_id"])
        date = pd.Timestamp(action["date"])
        source_unit = str(action["source_unit"])
        source_shift = normalize_shift(str(action["source_shift"]))
        target_unit = str(action["target_unit"])
        target_shift = normalize_shift(str(action["target_shift"]))

        if not self._is_valid_move(sid, source_unit, source_shift, target_unit, target_shift):
            return False

        mask = (
            (self.roster["staff_id"].astype(str) == sid)
            & (self.roster["date"] == date)
            & (self.roster["unit"] == source_unit)
            & (self.roster["shift"] == source_shift)
        )
        if not mask.any():
            return False
        idx = self.roster.index[mask][0]
        self.roster.loc[idx, "unit"] = target_unit
        self.roster.loc[idx, "shift"] = target_shift
        self.roster.loc[idx, "preference_score"] = self._preference(sid, date, target_shift)
        return True

    def _group_snapshot(
        self,
        date: pd.Timestamp,
        unit: str,
        shift: str,
    ) -> Dict[str, object]:
        """Return detailed metrics for one unit/shift group."""
        metrics = self._group_metrics(date, unit, shift)
        required = float(metrics.get("required", 0))
        assigned = float(metrics.get("assigned", 0))
        shortage = float(metrics.get("shortage", 0))
        skill_shortage = float(metrics.get("skill_shortage", 0))

        return {
            "date": normalize_date(date),
            "unit": str(unit),
            "shift": normalize_shift(shift),
            "required": required,
            "assigned": assigned,
            "shortage": shortage,
            "coverage": float(safe_divide(assigned, required)),
            "skill_shortage": skill_shortage,
            "skill_coverage": float(
                safe_divide(
                    sum(
                        float(detail["required"])
                        for detail in metrics.get("skill_details", {}).values()
                    ) - skill_shortage,
                    sum(
                        float(detail["required"])
                        for detail in metrics.get("skill_details", {}).values()
                    ),
                )
            ),
            "skill_details": metrics.get("skill_details", {}),
        }

    def _assignment_snapshot(
        self,
        staff_id: str,
        date: pd.Timestamp,
    ) -> List[Dict[str, object]]:
        """Return the staff member's assignments on the specified date."""
        rows = self.roster[
            (self.roster["staff_id"].astype(str) == str(staff_id))
            & (self.roster["date"] == pd.Timestamp(date))
        ]

        return [
            {
                "staff_id": str(row.staff_id),
                "date": normalize_date(row.date),
                "unit": str(row.unit),
                "shift": normalize_shift(row.shift),
                "shift_hours": float(row.shift_hours),
            }
            for row in rows.itertuples(index=False)
        ]

    def _reward_breakdown(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
        changed: bool,
        valid: bool,
    ) -> Dict[str, float]:
        """Return every reward component separately for transition diagnostics."""
        coverage_component = (
            self.config.coverage_reward
            * (after["coverage"] - before["coverage"])
        )
        skill_component = (
            self.config.skill_reward
            * (after["skill_coverage"] - before["skill_coverage"])
        )
        shortage_component = (
            -self.config.shortage_penalty
            * (after["shortage"] - before["shortage"])
        )
        skill_shortage_component = (
            -self.config.skill_shortage_penalty
            * (after["skill_shortage"] - before["skill_shortage"])
        )
        fairness_component = (
            -self.config.fairness_penalty
            * max(0.0, after["workload_gini"] - before["workload_gini"])
        )
        overtime_component = (
            -self.config.overtime_penalty
            * max(0.0, after["overtime"] - before["overtime"])
        )
        preference_component = (
            -self.config.preference_penalty
            * max(0.0, before["preference_score"] - after["preference_score"])
        )
        change_component = (
            -self.config.change_penalty if changed else 0.0
        )
        invalid_component = (
            -self.config.invalid_action_penalty if not valid else 0.0
        )

        total = (
            coverage_component
            + skill_component
            + shortage_component
            + skill_shortage_component
            + fairness_component
            + overtime_component
            + preference_component
            + change_component
            + invalid_component
        )

        return {
            "coverage": float(coverage_component),
            "skill_coverage": float(skill_component),
            "shortage": float(shortage_component),
            "skill_shortage": float(skill_shortage_component),
            "fairness": float(fairness_component),
            "overtime": float(overtime_component),
            "preference": float(preference_component),
            "change_penalty": float(change_component),
            "invalid_action": float(invalid_component),
            "total": float(total),
        }

    def _reward(self, before: Dict[str, float], after: Dict[str, float], changed: bool, valid: bool) -> float:
        return self._reward_breakdown(before, after, changed, valid)["total"]

        reward = 0.0
        reward += self.config.coverage_reward * (after["coverage"] - before["coverage"])
        reward += self.config.skill_reward * (after["skill_coverage"] - before["skill_coverage"])
        reward -= self.config.shortage_penalty * (after["shortage"] - before["shortage"])
        reward -= self.config.skill_shortage_penalty * (after["skill_shortage"] - before["skill_shortage"])
        reward -= self.config.fairness_penalty * max(0.0, after["workload_gini"] - before["workload_gini"])
        reward -= self.config.overtime_penalty * max(0.0, after["overtime"] - before["overtime"])
        reward -= self.config.preference_penalty * max(0.0, before["preference_score"] - after["preference_score"])
        if changed:
            reward -= self.config.change_penalty
        if not valid:
            reward -= self.config.invalid_action_penalty
        return float(reward)

    def step(self, action_index: int):
        if self.done:
            raise RuntimeError("Episode is already done. Call reset().")

        # The event stored in self.last_event belongs to the current action date.
        # Preserve it before advancing to the next date so diagnostics remain
        # temporally aligned with the action being evaluated.
        action_event = json.loads(json.dumps(self.last_event))

        before = self._all_metrics()
        self.candidates = self.get_valid_actions()

        valid_index = 0 <= int(action_index) < len(self.candidates)
        action = (
            self.candidates[int(action_index)]
            if valid_index
            else {"type": "invalid"}
        )

        changed = False
        valid = valid_index

        source_snapshot = None
        target_snapshot = None
        assignment_before = None
        assignment_after = None

        if valid_index and action.get("type") == "reassign":
            sid = str(action["staff_id"])
            action_date = pd.Timestamp(action["date"])
            source_unit = str(action["source_unit"])
            source_shift = normalize_shift(str(action["source_shift"]))
            target_unit = str(action["target_unit"])
            target_shift = normalize_shift(str(action["target_shift"]))

            source_snapshot = self._group_snapshot(
                action_date, source_unit, source_shift
            )
            target_snapshot = self._group_snapshot(
                action_date, target_unit, target_shift
            )
            assignment_before = self._assignment_snapshot(sid, action_date)

            changed = True
            valid = self._apply_action(action)

            assignment_after = self._assignment_snapshot(sid, action_date)

            if not valid:
                changed = False
        elif valid_index:
            # no-op is valid but does not modify the roster.
            valid = self._apply_action(action)

        after = self._all_metrics()

        # Capture the post-action local groups before the environment advances.
        if (
            valid_index
            and action.get("type") == "reassign"
            and source_snapshot is not None
            and target_snapshot is not None
        ):
            source_snapshot_after = self._group_snapshot(
                pd.Timestamp(action["date"]),
                str(action["source_unit"]),
                normalize_shift(str(action["source_shift"])),
            )
            target_snapshot_after = self._group_snapshot(
                pd.Timestamp(action["date"]),
                str(action["target_unit"]),
                normalize_shift(str(action["target_shift"])),
            )
        else:
            source_snapshot_after = None
            target_snapshot_after = None

        reward_breakdown = self._reward_breakdown(
            before, after, changed, valid
        )
        reward = reward_breakdown["total"]
        self.last_action = action

        # Action date is the date being evaluated, even though the environment
        # may now advance to the following date.
        action_date = pd.Timestamp(self.current_date)

        # Advance to the next day and create the next dynamic scenario.
        self.current_index += 1
        if self.current_index >= len(self.episode_dates):
            self.done = True
            next_state = self.get_state()
        else:
            self.current_date = self.episode_dates[self.current_index]
            self.current_shift = "Day"
            self.last_event = self._inject_event_for_current_date()
            self._remove_absent_assignments()
            self.candidates = self.get_valid_actions()
            next_state = self.get_state()

        info = {
            "date": normalize_date(action_date),
            "action": action,
            "valid_action": bool(valid),
            "changed": bool(changed),
            "before": before,
            "after": after,
            "reward_breakdown": reward_breakdown,
            "event": action_event,
            "candidate_count": len(self.candidates),
            "state_length": int(len(next_state)),
        }

        if source_snapshot is not None:
            info["source_before"] = source_snapshot
            info["source_after"] = source_snapshot_after

        if target_snapshot is not None:
            info["target_before"] = target_snapshot
            info["target_after"] = target_snapshot_after

        if assignment_before is not None:
            info["assignment_before"] = assignment_before

        if assignment_after is not None:
            info["assignment_after"] = assignment_after

        return next_state, reward, self.done, info



# -----------------------------------------------------------------------------
# Baseline loading and V3 scenario runner
# -----------------------------------------------------------------------------
def load_v2_baseline(dataset_path: str, roster_path: Optional[str] = None, time_limit: int = 120):
    manager = HospitalExcelDataManager(dataset_path)
    data = manager.load()
    if roster_path:
        roster = pd.read_csv(roster_path)
    else:
        scheduler = SkillAwareIntelligentStaffScheduler(V2SchedulingConfig(max_solver_seconds=time_limit))
        roster, _ = scheduler.generate_roster(
            data["staff"],
            data["demand_shift"],
            data["availability"],
            data["preferences"],
            data["staff_skills"],
        )
    return data, roster


def run_v3_smoke_test(
    dataset_path: str,
    roster_path: Optional[str] = None,
    output_dir: str = "v3_outputs",
    start_date: Optional[str] = None,
    time_limit: int = 120,
) -> Dict[str, str]:
    """Run deterministic V3 scenarios and a simple valid-action policy."""
    os.makedirs(output_dir, exist_ok=True)
    data, roster = load_v2_baseline(dataset_path, roster_path, time_limit)

    # Select staff actually present on the first episode date to make the
    # deterministic absence scenario meaningful.
    first_date = pd.Timestamp(start_date) if start_date else pd.Timestamp(data["demand_shift"]["date"].min())
    day_staff = roster[pd.to_datetime(roster["date"]) == first_date]["staff_id"].astype(str).unique().tolist()
    forced_absences = day_staff[: min(2, len(day_staff))]

    demand_rows = data["demand_shift"]
    target_rows = demand_rows[demand_rows["date"] == first_date].copy()
    if target_rows.empty:
        raise ValueError(f"No demand rows for {normalize_date(first_date)}")
    # Prefer a unit with a nonzero staffing requirement.
    target = target_rows.sort_values("required_staff", ascending=False).iloc[0]

    forced_events = [
        {
            "date": normalize_date(first_date),
            "absent_staff": forced_absences,
            "demand_surge": [
                {
                    "unit": str(target["unit"]),
                    "shift": normalize_shift(target["shift"]),
                    "extra_staff": 2,
                }
            ],
        }
    ]

    env = DynamicHospitalWorkforceEnvironment(
        data,
        roster,
        V3EnvironmentConfig(seed=42, episode_days=3),
    )
    state = env.reset(start_date=normalize_date(first_date), forced_events=forced_events)

    transitions = []
    done = False
    total_reward = 0.0
    step_no = 0
    while not done:
        # Baseline policy: choose the first actual reallocation candidate, if
        # one exists; otherwise choose no-op. This validates the environment
        # mechanics without claiming an RL result.
        action_index = 1 if len(env.candidates) > 1 else 0
        next_state, reward, done, info = env.step(action_index)
        total_reward += reward
        transitions.append(
            {
                "step": step_no,
                "date": info["date"],
                "action": info["action"],
                "valid_action": info["valid_action"],
                "changed": info["changed"],
                "reward": reward,
                "before": info["before"],
                "after": info["after"],
                "event": info["event"],
                "candidate_count": info["candidate_count"],
                "state_length": int(len(next_state)),
            }
        )
        state = next_state
        step_no += 1

    summary = {
        "version": "3.1",
        "description": "V3.1 dynamic staff-level hospital workforce environment validation smoke test",
        "start_date": normalize_date(first_date),
        "forced_absent_staff": forced_absences,
        "forced_demand_surge": {
            "unit": str(target["unit"]),
            "shift": normalize_shift(target["shift"]),
            "extra_staff": 2,
        },
        "episode_steps": step_no,
        "total_reward": float(total_reward),
        "final_state_length": int(len(state)),
        "final_metrics": env._all_metrics(),
    }

    transitions_path = os.path.join(output_dir, "v3_1_smoke_test_transitions.json")
    summary_path = os.path.join(output_dir, "v3_1_environment_summary.json")
    roster_path_out = os.path.join(output_dir, "v3_1_final_roster_after_smoke_test.csv")

    with open(transitions_path, "w", encoding="utf-8") as f:
        json.dump(transitions, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    env.roster.to_csv(roster_path_out, index=False)

    logger.info("V3 smoke test completed: %s", summary)
    return {
        "transitions": transitions_path,
        "summary": summary_path,
        "final_roster": roster_path_out,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Version 3.1 dynamic hospital workforce environment validation smoke test.")
    parser.add_argument("--dataset", required=True, help="Path to hospital_hrm_dataset.xlsx")
    parser.add_argument("--roster", default=None, help="Optional V2 roster CSV. If omitted, V2 solves a baseline first.")
    parser.add_argument("--output", default="v3_outputs", help="V3 output directory")
    parser.add_argument("--start-date", default=None, help="Episode start date, e.g. 2025-11-03")
    parser.add_argument("--time-limit", type=int, default=120, help="V2 CP-SAT time limit if a roster must be generated")
    args = parser.parse_args()

    outputs = run_v3_smoke_test(
        dataset_path=args.dataset,
        roster_path=args.roster,
        output_dir=args.output,
        start_date=args.start_date,
        time_limit=args.time_limit,
    )
    print("\nVersion 3.1 outputs:")
    for key, path in outputs.items():
        print(f"- {key}: {path}")
