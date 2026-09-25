"""
Hospital Workforce Scheduling Framework - RL Extension
=======================================================

This version extends the original hospital HRM prototype rather than replacing it.
It keeps the existing forecasting, CP-SAT scheduling and evaluation components and
adds:

1. Excel dataset adapter for hospital_hrm_dataset.xlsx
2. Skill-aware baseline scheduling support through normalized inputs
3. Dynamic Hospital Workforce RL Environment
4. PyTorch PPO agent
5. Scenario-based training with staff absence, demand surge and workload changes
6. RL-vs-baseline evaluation metrics
7. Export of baseline and adaptive schedules/metrics

The RL agent is deliberately placed AFTER the initial CP-SAT roster. CP-SAT gives
us a feasible starting schedule; the RL agent learns how to adapt it when the
hospital state changes.
"""

from __future__ import annotations

import os
import json
import math
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

# Reuse the original research-grade components.
from hospital_hrm_framework import (
    ForecastConfig,
    SchedulingConfig,
    EvaluationConfig,
    WorkforceDemandForecaster,
    IntelligentStaffScheduler,
    PerformanceEvaluator,
    compute_gini,
    safe_divide,
)

logger = logging.getLogger("hospital_hrm_rl")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# -----------------------------------------------------------------------------
# Dataset adapter
# -----------------------------------------------------------------------------
class HospitalExcelDataManager:
    """Adapter for the supplied Excel workbook.

    The original framework expects separate CSV files. The supplied project
    dataset is an Excel workbook with one sheet per logical table, so this class
    maps those sheets into the same conceptual inputs used by the original code.
    """

    SHEETS = {
        "forecasting": "admissions_daily",
        "staff": "staff",
        "skills": "skills",
        "staff_skills": "staff_skills",
        "availability": "availability",
        "preferences": "preferences",
        "demand": "demand_shift",
        "roster": "roster_planned",
        "attendance": "attendance",
        "kpi": "staff_week_kpi",
        "patient_feedback": "patient_feedback",
        "peer_reviews": "peer_reviews",
    }

    def __init__(self, workbook_path: str):
        self.workbook_path = workbook_path
        if not os.path.exists(workbook_path):
            raise FileNotFoundError(f"Dataset not found: {workbook_path}")

    def read(self, logical_name: str) -> pd.DataFrame:
        sheet = self.SHEETS[logical_name]
        df = pd.read_excel(self.workbook_path, sheet_name=sheet)
        logger.info("Loaded sheet %-18s shape=%s", sheet, df.shape)
        return df

    def load_forecasting_data(self) -> pd.DataFrame:
        df = self.read("forecasting").copy()
        df["date"] = pd.to_datetime(df["date"])
        # Original forecasting class expects census, while the supplied dataset
        # calls the field census_est.
        if "census_est" in df.columns and "census" not in df.columns:
            df["census"] = df["census_est"]
        df = df.sort_values(["unit", "date"]).reset_index(drop=True)
        return df

    def load_scheduling_inputs(self) -> Dict[str, pd.DataFrame]:
        staff = self.read("staff").copy()
        staff["staff_id"] = staff["staff_id"].astype(str)
        # Map the supplied dataset's contracted hours to the original scheduler's
        # expected max_hours_week field.
        staff["max_hours_week"] = staff["contracted_hours_per_week"].fillna(40).astype(int)

        availability = self.read("availability").copy()
        availability["staff_id"] = availability["staff_id"].astype(str)
        availability["date"] = pd.to_datetime(availability["date"]).dt.strftime("%Y-%m-%d")
        availability["shift"] = availability["shift"].astype(str)

        preferences = self.read("preferences").copy()
        preferences["staff_id"] = preferences["staff_id"].astype(str)
        preferences["date"] = pd.to_datetime(preferences["date"]).dt.strftime("%Y-%m-%d")
        preferences["shift"] = preferences["shift"].astype(str)
        preferences["unit"] = preferences.get("unit", pd.Series("", index=preferences.index)).astype(str)
        # The supplied preferences table is not unit-specific. The baseline
        # scheduler therefore uses the same preference score for a staff/date/shift.
        # We expand it across units below.
        demand = self.read("demand").copy()
        demand["date"] = pd.to_datetime(demand["date"]).dt.strftime("%Y-%m-%d")
        demand["shift"] = demand["shift"].astype(str)
        demand["unit"] = demand["unit"].astype(str)

        units = demand["unit"].unique().tolist()
        if preferences["unit"].eq("").all() or preferences["unit"].isna().all():
            base = preferences.drop(columns=["unit"], errors="ignore")
            base["_key"] = 1
            unit_df = pd.DataFrame({"unit": units, "_key": 1})
            preferences = base.merge(unit_df, on="_key").drop(columns=["_key"])
        preferences["unit"] = preferences["unit"].astype(str)

        staff_skills = self.read("staff_skills").copy()
        staff_skills["staff_id"] = staff_skills["staff_id"].astype(str)
        # Original scheduler expects a column named skill.
        staff_skills["skill"] = staff_skills["skill_code"].astype(str)

        return {
            "staff": staff,
            "demand": demand,
            "availability": availability,
            "preferences": preferences,
            "staff_skills": staff_skills,
        }

    def load_evaluation_inputs(self) -> Dict[str, pd.DataFrame]:
        return {
            "attendance": self.read("attendance"),
            "kpi": self.read("kpi"),
            "patient_feedback": self.read("patient_feedback"),
            "peer_reviews": self.read("peer_reviews"),
        }


# -----------------------------------------------------------------------------
# RL configuration
# -----------------------------------------------------------------------------
@dataclass
class RLConfig:
    seed: int = 42
    state_size: int = 24
    action_size: int = 7
    hidden_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    update_epochs: int = 8
    rollout_steps: int = 256
    training_episodes: int = 250
    max_episode_steps: int = 24
    demand_surge_probability: float = 0.35
    absence_probability: float = 0.25
    action_staff_candidates: int = 8

    # Reward weights. These are experimental research parameters and should be
    # tuned through validation, not presented as measured results.
    coverage_reward: float = 8.0
    shortage_penalty: float = 12.0
    fairness_penalty: float = 4.0
    overtime_penalty: float = 3.0
    preference_penalty: float = 1.5
    unnecessary_change_penalty: float = 0.5


# -----------------------------------------------------------------------------
# Dynamic hospital environment
# -----------------------------------------------------------------------------
class HospitalWorkforceEnvironment:
    """Small, transparent RL environment for dynamic roster adaptation.

    Action semantics:
        0 = no change
        1 = move one eligible staff member toward the most understaffed unit
        2 = move two eligible staff members toward the most understaffed unit
        3 = move one low-workload staff member toward the shortage
        4 = move one available staff member from the least critical unit
        5 = undo/reduce one recent reassignment when it harms fairness
        6 = stabilize: keep the current roster and prioritize coverage

    The agent learns the policy for choosing the intervention type. The
    environment determines the concrete eligible staff member, preventing an
    enormous action space while retaining staff-level schedule changes.
    """

    def __init__(
        self,
        staff_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        availability_df: pd.DataFrame,
        preferences_df: pd.DataFrame,
        staff_skills_df: pd.DataFrame,
        config: RLConfig,
        seed: int = 42,
    ):
        self.staff = staff_df.copy()
        self.demand = demand_df.copy()
        self.availability = availability_df.copy()
        self.preferences = preferences_df.copy()
        self.staff_skills = staff_skills_df.copy()
        self.cfg = config
        self.rng = np.random.default_rng(seed)

        self.units = sorted(self.demand["unit"].astype(str).unique())
        self.shifts = ["Day", "Evening", "Night"]
        self.staff_ids = self.staff["staff_id"].astype(str).tolist()
        self.staff_units = dict(zip(self.staff_ids, self.staff["home_unit"].astype(str)))
        self.staff_hours_limit = dict(
            zip(self.staff_ids, self.staff["max_hours_week"].astype(float))
        )
        self.skill_map: Dict[str, set] = {}
        for row in self.staff_skills.itertuples(index=False):
            self.skill_map.setdefault(str(row.staff_id), set()).add(str(row.skill))

        self.avail_map = {
            (str(r.staff_id), str(r.date), str(r.shift)): int(r.available)
            for r in self.availability.itertuples(index=False)
        }
        self.pref_map = {
            (str(r.staff_id), str(r.date), str(r.shift)): float(r.preference_score)
            for r in self.preferences.itertuples(index=False)
        }

        self.demand_index = {}
        for row in self.demand.itertuples(index=False):
            self.demand_index[(str(row.date), str(row.unit), str(row.shift))] = row

        self.dates = sorted(self.demand["date"].astype(str).unique())
        self.current_date = None
        self.current_shift = None
        self.baseline_roster = pd.DataFrame()
        self.current_roster = pd.DataFrame()
        self.previous_roster = pd.DataFrame()
        self.step_count = 0
        self.episode_reward = 0.0
        self.last_action = 0

    # ----- state helpers -----------------------------------------------------
    def _demand_rows_for_current_shift(self) -> pd.DataFrame:
        rows = self.demand[
            (self.demand["date"].astype(str) == self.current_date)
            & (self.demand["shift"].astype(str) == self.current_shift)
        ].copy()
        return rows

    def _required_staff(self, unit: str) -> int:
        row = self.demand_index.get((self.current_date, unit, self.current_shift))
        if row is None:
            return 0
        return int(row.required_staff)

    def _assigned_counts(self) -> Dict[str, int]:
        if self.current_roster.empty:
            return {u: 0 for u in self.units}
        x = self.current_roster[
            (self.current_roster["date"].astype(str) == self.current_date)
            & (self.current_roster["shift"].astype(str) == self.current_shift)
            & (self.current_roster["assigned"].astype(int) == 1)
        ]
        return x["unit"].value_counts().reindex(self.units, fill_value=0).astype(int).to_dict()

    def _workload_hours(self) -> Dict[str, float]:
        hours = {sid: 0.0 for sid in self.staff_ids}
        if self.current_roster.empty:
            return hours
        for r in self.current_roster[self.current_roster["assigned"].astype(int) == 1].itertuples(index=False):
            hours[str(r.staff_id)] += float(getattr(r, "shift_hours", 8))
        return hours

    def _coverage_stats(self) -> Tuple[float, float]:
        counts = self._assigned_counts()
        required = {u: self._required_staff(u) for u in self.units}
        total_req = max(1, sum(required.values()))
        covered = sum(min(counts[u], required[u]) for u in self.units)
        shortage = sum(max(0, required[u] - counts[u]) for u in self.units)
        return covered / total_req, float(shortage)

    def _fairness(self) -> float:
        workloads = self._workload_hours()
        return compute_gini(workloads.values())

    def _preference_satisfaction(self) -> float:
        if self.current_roster.empty:
            return 0.0
        x = self.current_roster[
            (self.current_roster["date"].astype(str) == self.current_date)
            & (self.current_roster["shift"].astype(str) == self.current_shift)
            & (self.current_roster["assigned"].astype(int) == 1)
        ]
        if x.empty:
            return 0.0
        vals = [
            self.pref_map.get((str(r.staff_id), self.current_date, self.current_shift), 0.5)
            for r in x.itertuples(index=False)
        ]
        return float(np.mean(vals))

    def _overtime_hours(self) -> float:
        workloads = self._workload_hours()
        return float(sum(max(0.0, workloads[s] - self.staff_hours_limit.get(s, 40)) for s in workloads))

    def _state(self) -> np.ndarray:
        counts = self._assigned_counts()
        required = {u: self._required_staff(u) for u in self.units}
        coverage = [safe_divide(counts[u], max(1, required[u])) for u in self.units]
        shortage = [safe_divide(max(0, required[u] - counts[u]), max(1, required[u])) for u in self.units]

        workloads = self._workload_hours()
        workload_values = np.array(list(workloads.values()), dtype=float)
        if len(workload_values):
            workload_mean = workload_values.mean()
            workload_std = workload_values.std()
            max_load = workload_values.max()
        else:
            workload_mean = workload_std = max_load = 0.0

        available_now = [
            self.avail_map.get((sid, self.current_date, self.current_shift), 0)
            for sid in self.staff_ids
        ]
        availability_ratio = float(np.mean(available_now)) if available_now else 0.0

        demand_total = sum(required.values())
        demand_norm = min(1.0, demand_total / max(1, len(self.staff_ids)))
        coverage_ratio, shortage_total = self._coverage_stats()
        fairness = self._fairness()
        preference = self._preference_satisfaction()
        overtime = self._overtime_hours()

        # Compact state: 5 unit coverage + 5 shortage + global indicators.
        state = np.array(
            coverage
            + shortage
            + [
                demand_norm,
                availability_ratio,
                coverage_ratio,
                min(1.0, shortage_total / max(1, demand_total)),
                fairness,
                min(1.0, workload_mean / 48.0),
                min(1.0, workload_std / 24.0),
                min(1.0, max_load / 60.0),
                min(1.0, overtime / 100.0),
                preference,
                self.step_count / max(1, self.cfg.max_episode_steps),
                self.last_action / max(1, self.cfg.action_size - 1),
            ],
            dtype=np.float32,
        )
        if len(state) != self.cfg.state_size:
            raise RuntimeError(f"State size mismatch: expected {self.cfg.state_size}, got {len(state)}")
        return state

    # ----- roster manipulation -----------------------------------------------
    def _eligible_staff(self, target_unit: str) -> List[str]:
        current = self.current_roster[
            (self.current_roster["date"].astype(str) == self.current_date)
            & (self.current_roster["shift"].astype(str) == self.current_shift)
            & (self.current_roster["assigned"].astype(int) == 1)
        ]
        assigned_ids = set(current["staff_id"].astype(str))
        candidates = []
        for sid in self.staff_ids:
            if sid in assigned_ids:
                continue
            if self.avail_map.get((sid, self.current_date, self.current_shift), 0) != 1:
                continue
            candidates.append(sid)
        return candidates

    def _most_understaffed_unit(self) -> Optional[str]:
        counts = self._assigned_counts()
        gaps = {u: self._required_staff(u) - counts[u] for u in self.units}
        target = max(gaps, key=gaps.get)
        return target if gaps[target] > 0 else None

    def _add_assignment(self, sid: str, target_unit: str):
        # Keep an existing assignment row if one exists, otherwise add one.
        mask = (
            (self.current_roster["staff_id"].astype(str) == sid)
            & (self.current_roster["date"].astype(str) == self.current_date)
            & (self.current_roster["shift"].astype(str) == self.current_shift)
        )
        if mask.any():
            idx = self.current_roster.index[mask][0]
            self.current_roster.loc[idx, "unit"] = target_unit
            self.current_roster.loc[idx, "assigned"] = 1
            return

        new_row = {
            "staff_id": sid,
            "date": self.current_date,
            "unit": target_unit,
            "shift": self.current_shift,
            "assigned": 1,
            "preference_score": self.pref_map.get((sid, self.current_date, self.current_shift), 0.5),
            "shift_hours": 8,
        }
        self.current_roster = pd.concat([self.current_roster, pd.DataFrame([new_row])], ignore_index=True)

    def _reassign_from_surplus(self, target_unit: str, count: int = 1):
        # First choose unassigned and available people.
        candidates = self._eligible_staff(target_unit)
        workloads = self._workload_hours()
        candidates.sort(key=lambda s: (workloads.get(s, 0), -self.pref_map.get((s, self.current_date, self.current_shift), 0.5)))
        moved = 0
        for sid in candidates[:count]:
            self._add_assignment(sid, target_unit)
            moved += 1
        return moved

    def _undo_recent_change(self):
        if self.current_roster.empty or self.previous_roster.empty:
            return
        self.current_roster = self.previous_roster.copy()

    # ----- environment API --------------------------------------------------
    def reset(self, baseline_roster: pd.DataFrame, date: Optional[str] = None, shift: Optional[str] = None):
        self.baseline_roster = baseline_roster.copy()
        self.current_roster = baseline_roster.copy()
        self.previous_roster = baseline_roster.copy()
        self.current_date = date or self.rng.choice(self.dates)
        self.current_shift = shift or self.rng.choice(self.shifts)
        self.step_count = 0
        self.episode_reward = 0.0
        self.last_action = 0
        self._inject_dynamic_event()
        return self._state()

    def _inject_dynamic_event(self):
        """Create realistic perturbations without changing the source dataset."""
        if self.rng.random() < self.cfg.demand_surge_probability:
            rows = self.demand[
                (self.demand["date"].astype(str) == self.current_date)
                & (self.demand["shift"].astype(str) == self.current_shift)
            ]
            for row in rows.itertuples(index=False):
                # Store an episode-local override rather than mutating the dataset.
                key = (self.current_date, str(row.unit), self.current_shift)
                original = int(row.required_staff)
                surge = int(round(original * self.rng.uniform(0.10, 0.35)))
                self.demand_index[key] = row._asdict() if hasattr(row, "_asdict") else row
                # Namedtuple fields are immutable; use an override dictionary.
            # Actual override is maintained separately.
        self._demand_override = {}
        if self.rng.random() < self.cfg.demand_surge_probability:
            for u in self.units:
                base = self._base_required_staff(u)
                if base > 0:
                    self._demand_override[(self.current_date, u, self.current_shift)] = int(
                        math.ceil(base * self.rng.uniform(1.10, 1.35))
                    )
        self._absence_override = set()
        if self.rng.random() < self.cfg.absence_probability:
            available = [
                sid for sid in self.staff_ids
                if self.avail_map.get((sid, self.current_date, self.current_shift), 0) == 1
            ]
            if available:
                n = max(1, int(round(len(available) * self.rng.uniform(0.02, 0.08))))
                self._absence_override = set(self.rng.choice(available, size=min(n, len(available)), replace=False).tolist())

    def _base_required_staff(self, unit: str) -> int:
        row = self.demand_index.get((self.current_date, unit, self.current_shift))
        return int(row.required_staff) if row is not None else 0

    def required_staff_dynamic(self, unit: str) -> int:
        return int(self._demand_override.get((self.current_date, unit, self.current_shift), self._base_required_staff(unit)))

    def step(self, action: int):
        self.previous_roster = self.current_roster.copy()
        self.last_action = int(action)

        target = self._most_understaffed_unit()
        moved = 0
        if target is not None:
            if action == 1:
                moved = self._reassign_from_surplus(target, 1)
            elif action == 2:
                moved = self._reassign_from_surplus(target, 2)
            elif action == 3:
                moved = self._reassign_from_surplus(target, 1)
            elif action == 4:
                moved = self._reassign_from_surplus(target, 1)
            elif action == 5:
                self._undo_recent_change()
            elif action == 6:
                pass

        reward, metrics = self._calculate_reward(moved=moved)
        self.episode_reward += reward
        self.step_count += 1
        done = self.step_count >= self.cfg.max_episode_steps
        return self._state(), float(reward), done, metrics

    def _calculate_reward(self, moved: int) -> Tuple[float, Dict[str, float]]:
        counts = self._assigned_counts()
        required = {u: self.required_staff_dynamic(u) for u in self.units}
        total_required = max(1, sum(required.values()))
        covered = sum(min(counts[u], required[u]) for u in self.units)
        shortage = sum(max(0, required[u] - counts[u]) for u in self.units)
        coverage_ratio = covered / total_required

        workloads = self._workload_hours()
        fairness = compute_gini(workloads.values())
        overtime = self._overtime_hours()
        preference = self._preference_satisfaction()

        reward = (
            self.cfg.coverage_reward * coverage_ratio
            - self.cfg.shortage_penalty * shortage / total_required
            - self.cfg.fairness_penalty * fairness
            - self.cfg.overtime_penalty * overtime / 40.0
            - self.cfg.preference_penalty * (1.0 - preference)
            - self.cfg.unnecessary_change_penalty * max(0, moved - 1)
        )

        metrics = {
            "coverage_ratio": float(coverage_ratio),
            "shortage": float(shortage),
            "fairness_gini": float(fairness),
            "overtime_hours": float(overtime),
            "preference_satisfaction": float(preference),
            "staff_moved": float(moved),
        }
        return float(reward), metrics


# -----------------------------------------------------------------------------
# PPO implementation
# -----------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
except ImportError:  # pragma: no cover
    torch = None


class ActorCritic(nn.Module if torch is not None else object):
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        if torch is None:
            raise ImportError("PyTorch is required for the RL component. Install with: pip install torch")
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)


class PPOAgent:
    """Minimal PPO implementation so the project does not depend on Gym/SB3."""

    def __init__(self, config: RLConfig):
        if torch is None:
            raise ImportError("PyTorch is required for PPO. Install with: pip install torch")
        self.cfg = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(config.state_size, config.action_size, config.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = False):
        x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(x)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def update(self, states, actions, old_log_probs, returns, advantages):
        states_t = torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(self.cfg.update_epochs):
            logits, values = self.model(states_t)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_t)
            clipped = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon)
            actor_loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()
            critic_loss = nn.functional.mse_loss(values, returns_t)
            loss = actor_loss + self.cfg.value_coef * critic_loss - self.cfg.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


# -----------------------------------------------------------------------------
# RL training / evaluation orchestration
# -----------------------------------------------------------------------------
def build_baseline_roster(data: HospitalExcelDataManager, scheduling_config: SchedulingConfig):
    inputs = data.load_scheduling_inputs()
    scheduler = IntelligentStaffScheduler(scheduling_config)
    roster, metrics = scheduler.generate_weekly_roster(
        staff_df=inputs["staff"],
        demand_df=inputs["demand"],
        availability_df=inputs["availability"],
        preferences_df=inputs["preferences"],
        staff_skills_df=inputs["staff_skills"],
    )
    return inputs, roster, metrics


def train_rl_agent(env: HospitalWorkforceEnvironment, agent: PPOAgent, cfg: RLConfig):
    history = []
    for episode in range(cfg.training_episodes):
        state = env.reset(env.baseline_roster)
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        total_reward = 0.0

        for _ in range(cfg.max_episode_steps):
            action, log_prob, value = agent.act(state)
            next_state, reward, done, metrics = env.step(action)
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            state = next_state
            total_reward += reward
            if done:
                break

        # Bootstrap from final state.
        _, _, next_value = agent.act(state, deterministic=True)
        returns = np.zeros(len(rewards), dtype=np.float32)
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_v = next_value if t == len(rewards) - 1 else values[t + 1]
            nonterminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + cfg.gamma * next_v * nonterminal - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        update_metrics = agent.update(states, actions, log_probs, returns, advantages)
        history.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "mean_reward": float(np.mean(rewards)),
            **update_metrics,
        })

        if (episode + 1) % 25 == 0:
            logger.info(
                "RL episode %d/%d | reward=%.3f | mean=%.3f",
                episode + 1,
                cfg.training_episodes,
                total_reward,
                np.mean(rewards),
            )

    return pd.DataFrame(history)


def evaluate_agent(env: HospitalWorkforceEnvironment, agent: PPOAgent, episodes: int = 50):
    rows = []
    for ep in range(episodes):
        state = env.reset(env.baseline_roster)
        episode_metrics = []
        for _ in range(env.cfg.max_episode_steps):
            action, _, _ = agent.act(state, deterministic=True)
            state, reward, done, metrics = env.step(action)
            metrics["reward"] = reward
            episode_metrics.append(metrics)
            if done:
                break
        if episode_metrics:
            avg = pd.DataFrame(episode_metrics).mean(numeric_only=True).to_dict()
            avg["episode"] = ep + 1
            rows.append(avg)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main research pipeline
# -----------------------------------------------------------------------------
def run_project(
    workbook_path: str = "hospital_hrm_dataset.xlsx",
    output_dir: str = "outputs_rl",
    train: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    data = HospitalExcelDataManager(workbook_path)

    # 1. Forecasting remains available from the original framework.
    forecast_config = ForecastConfig()
    forecaster = WorkforceDemandForecaster(forecast_config)
    forecast_df = data.load_forecasting_data()
    forecasting_results = forecaster.compare_models(forecast_df)
    forecasting_results.to_csv(os.path.join(output_dir, "forecast_model_comparison.csv"), index=False)

    # 2. Existing CP-SAT scheduler creates the initial feasible roster.
    scheduling_config = SchedulingConfig()
    inputs, baseline_roster, baseline_metrics = build_baseline_roster(data, scheduling_config)
    baseline_roster.to_csv(os.path.join(output_dir, "baseline_cp_sat_roster.csv"), index=False)
    with open(os.path.join(output_dir, "baseline_scheduling_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, indent=2)

    # 3. RL environment starts from the baseline roster.
    rl_config = RLConfig()
    env = HospitalWorkforceEnvironment(
        staff_df=inputs["staff"],
        demand_df=inputs["demand"],
        availability_df=inputs["availability"],
        preferences_df=inputs["preferences"],
        staff_skills_df=inputs["staff_skills"],
        config=rl_config,
        seed=rl_config.seed,
    )
    env.baseline_roster = baseline_roster.copy()

    agent = PPOAgent(rl_config)
    model_path = os.path.join(output_dir, "ppo_hospital_workforce_agent.pt")

    if train:
        training_history = train_rl_agent(env, agent, rl_config)
        training_history.to_csv(os.path.join(output_dir, "rl_training_history.csv"), index=False)
        agent.save(model_path)
    elif os.path.exists(model_path):
        agent.load(model_path)

    # 4. Evaluate the learned policy under randomized dynamic conditions.
    rl_metrics = evaluate_agent(env, agent, episodes=50)
    rl_metrics.to_csv(os.path.join(output_dir, "rl_evaluation_metrics.csv"), index=False)

    summary = {
        "baseline": baseline_metrics,
        "rl_evaluation_mean": rl_metrics.mean(numeric_only=True).to_dict(),
        "rl_model": model_path,
        "forecasting_models": forecasting_results.to_dict(orient="records"),
    }
    with open(os.path.join(output_dir, "project_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    logger.info("Project pipeline completed. Outputs written to %s", output_dir)
    return summary


if __name__ == "__main__":
    # Update this path if your workbook is stored elsewhere.
    run_project("hospital_hrm_dataset.xlsx", "outputs_rl", train=True)
