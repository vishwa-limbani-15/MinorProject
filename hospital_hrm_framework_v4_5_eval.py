"""
Hospital HRM Framework - Version 4.5
====================================

Builds on V2 (skill-aware CP-SAT baseline) and V3 (dynamic staff-level
environment).  V4.4 weaknesses that V4.5 addresses
------------------------------------------------------------------------
1. Fixed action *indices* meant different staff moves in every state, so PPO
   could not learn what an index means  ->  V4.5 uses a candidate-scoring
   policy (shared network scores every candidate from its own feature vector).
2. One move per day could not repair "+2 surge and 2 absences"  ->  V4.5 allows
   up to ``moves_per_day`` sequential moves; the agent ends the day with no-op.
3. Reassigning staff between units cannot create capacity  ->  V4.5 adds a
   ``call_in`` action (bring in an off-duty, available, rested, within-contract
   employee) next to the V3 ``reassign`` action.
4. The V4.4 reward double-counted event-local gains, was unnormalised and not
   potential-based  ->  V4.5 reward = potential difference + move cost.
5. PPO saw ~7 samples per update  ->  V4.5 batches several episodes per update,
   normalises observations, uses correct GAE, randomised disruption curricula
   and a decaying behaviour-cloning term toward a measured greedy teacher.
6. PPO was only a 0.001 tie-breaker  ->  V4.5 evaluates the learned policy
   itself (with a measured safety gate) and ablates it against greedy, no-op,
   V4.4 and an exact CP-SAT day-repair arm.
7. Evaluation had no statistics  ->  paired bootstrap CIs + Wilcoxon, two
   disruption suites, end-of-day metrics.

All operational rules (availability, request-off, rest, contract hours, one
shift per day, skill gating) stay hard constraints.  Research prototype, not a
clinical production system.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from hospital_hrm_framework_v2 import compute_gini, normalize_shift
from hospital_hrm_framework_v3 import (
    DynamicHospitalWorkforceEnvironment,
    V3EnvironmentConfig,
    load_v2_baseline,
)
from dataclasses import dataclass as _dc  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("hospital_hrm_v4_5")

FEAT_DIM = 15
REPAIR = -1
_EMPTY: frozenset = frozenset()


def _ival(x) -> int:
    try:
        return 0 if pd.isna(x) else int(x)
    except Exception:
        return 0


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class V45Config(V3EnvironmentConfig):
    max_candidates: int = 48
    episode_days: int = 7
    moves_per_day: int = 6
    allow_call_in: bool = True
    call_in_per_target: int = 6
    max_targets: int = 8                # shortage groups considered per step (bounds cost on big hospitals)
    overtime_allowance: float = 0.0     # extra weekly hours a call-in may use (0 = hard V2 cap)

    # Potential Phi = -(w_short*shortage + w_skill*skill_shortage + w_overtime*overtime/8
    #                   + w_gini*gini + w_pref*(1-mean_pref))
    w_short: float = 1.0
    w_skill: float = 0.7
    w_overtime: float = 0.5
    w_gini: float = 8.0
    w_pref: float = 0.5
    cost_reassign: float = 0.05
    cost_call_in: float = 0.10
    invalid_penalty: float = 1.0


@dataclass
class V45PPOConfig:
    seed: int = 42
    updates: int = 60
    episodes_per_update: int = 8
    bc_updates: int = 6
    bc_epochs: int = 6
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    hidden_size: int = 128
    bc_coef_start: float = 0.5
    bc_coef_end: float = 0.05
    target_kl: float = 0.03


# =============================================================================
# Environment
# =============================================================================
class AdaptiveWorkforceEnv(DynamicHospitalWorkforceEnvironment):
    """V3 environment, made fast and extended (call-in, multi-move, new reward).

    One *decision step* is one action.  The day only closes when the agent
    chooses no-op (or an invalid index); then the next day's event is injected.
    """

    SHIFTS = ["Day", "Evening", "Night"]

    # ------------------------------------------------------------------ init
    def __init__(self, data, baseline_roster, config: Optional[V45Config] = None):
        config = config or V45Config()
        super().__init__(data, baseline_roster, config)
        self.cfg = config
        self.moves_today = 0
        self.n_changes = 0
        self._ctx_cache = None
        self._suspend = False
        self.day_open: Dict[str, float] = {}
        self._forb = {(normalize_shift(a), normalize_shift(b)) for a, b in config.forbidden_pairs}
        self._units = sorted(self.demand["unit"].unique())

        av = self.availability.drop_duplicates(["staff_id", "date", "shift"])
        self._avail_map = {
            (s, d, sh): int(a) == 1
            for s, d, sh, a in zip(av["staff_id"], av["date"], av["shift"], av["available"])
        }
        pf = self.preferences
        self._off_set = {
            (s, d, sh)
            for s, d, sh, o in zip(pf["staff_id"], pf["date"], pf["shift"], pf["request_off"].fillna(0))
            if int(o) == 1
        }
        pfd = pf.drop_duplicates(["staff_id", "date", "shift"])
        self._pref_map: Dict[Tuple, float] = {}
        for s, d, sh, p in zip(pfd["staff_id"], pfd["date"], pfd["shift"], pfd["preference_score"]):
            v = float(p) if pd.notna(p) else 0.5
            self._pref_map[(s, d, sh)] = v if v != 0.0 else 0.5   # V3 quirk: 0 -> 0.5

        K = config.max_candidates + 1
        self.cand_feats = np.zeros((K, FEAT_DIM), dtype=np.float32)
        self.cand_mask = np.zeros(K, dtype=bool)
        self.cand_mask[0] = True
        self.teacher_idx = 0

    # ------------------------------------------------------- fast V3 lookups
    def _availability(self, staff_id, date, shift):
        return self._avail_map.get((str(staff_id), pd.Timestamp(date), normalize_shift(shift)), False)

    def _request_off(self, staff_id, date, shift):
        return (str(staff_id), pd.Timestamp(date), normalize_shift(shift)) in self._off_set

    def _preference(self, staff_id, date, shift):
        return self._pref_map.get((str(staff_id), pd.Timestamp(date), normalize_shift(shift)), 0.5)

    def _inject_event_for_current_date(self):
        ev = super()._inject_event_for_current_date()
        self._ctx_cache = None
        return ev

    def _remove_absent_assignments(self):
        super()._remove_absent_assignments()
        self._ctx_cache = None

    # --------------------------------------------------------------- context
    @staticmethod
    def _week_bounds(date: pd.Timestamp):
        d = pd.Timestamp(date).normalize()
        lo = d - pd.Timedelta(days=d.weekday())
        return lo, lo + pd.Timedelta(days=7)

    @staticmethod
    def _gap(ids: List[str], req: int, skills: Dict[str, int], skill_map) -> Tuple[int, int]:
        short = max(0, req - len(ids))
        ss = 0
        for code, n in skills.items():
            q = 0
            for s in ids:
                if code in skill_map.get(s, _EMPTY):
                    q += 1
            ss += max(0, n - q)
        return short, ss

    def _ctx(self) -> Dict:
        if self._ctx_cache is not None:
            return self._ctx_cache
        date = pd.Timestamp(self.current_date)
        r = self.roster
        today = r[r["date"] == date]
        groups: Dict[Tuple[str, str], Dict] = {}
        for rec in self.current_demand[self.current_demand["date"] == date].to_dict("records"):
            key = (str(rec["unit"]), normalize_shift(rec["shift"]))
            skills = {}
            for col in self.skill_columns:
                n = _ival(rec[col])
                if n > 0:
                    skills[col[len("min_"):]] = n
            groups[key] = {"req": _ival(rec["required_staff"]), "skills": skills, "ids": []}
        assign: Dict[str, Tuple[str, str]] = {}
        for sid, u, sh in zip(today["staff_id"], today["unit"], today["shift"]):
            key = (str(u), normalize_shift(sh))
            assign[str(sid)] = key
            if key in groups:
                groups[key]["ids"].append(str(sid))

        tot_req = tot_asg = tot_short = tot_sreq = tot_sshort = 0
        for key, g in groups.items():
            short, ss = self._gap(g["ids"], g["req"], g["skills"], self.skill_map)
            needed = [
                c for c, n in g["skills"].items()
                if sum(c in self.skill_map.get(s, _EMPTY) for s in g["ids"]) < n
            ]
            g.update(short=short, ss=ss, needed=needed)
            tot_req += g["req"]; tot_asg += len(g["ids"]); tot_short += short
            tot_sreq += sum(g["skills"].values()); tot_sshort += ss

        lo, hi = self._week_bounds(date)
        wk_df = r[(r["date"] >= lo) & (r["date"] < hi)]
        wk = {str(k): float(v) for k, v in wk_df.groupby("staff_id")["shift_hours"].sum().items()}
        order = list(self.staff_hours.keys())
        w_vec = np.array([wk.get(s, 0.0) for s in order], dtype=float)
        contracts = np.array([self.staff_hours[s] for s in order], dtype=float)
        gini = compute_gini(w_vec)
        overtime = float(np.maximum(0.0, w_vec - contracts).sum())
        pref_sum = sum(self._preference(s, date, k[1]) for s, k in assign.items())
        n_today = len(assign)
        pref_mean = pref_sum / n_today if n_today else 0.5

        prev: Dict[str, set] = {}
        nxt: Dict[str, set] = {}
        near = r[(r["date"] == date - pd.Timedelta(days=1)) | (r["date"] == date + pd.Timedelta(days=1))]
        for sid, d, sh in zip(near["staff_id"], near["date"], near["shift"]):
            (prev if d < date else nxt).setdefault(str(sid), set()).add(normalize_shift(sh))

        metrics = {
            "required": float(tot_req), "assigned": float(tot_asg), "shortage": float(tot_short),
            "coverage": tot_asg / tot_req if tot_req else 0.0,
            "skill_required": float(tot_sreq), "skill_shortage": float(tot_sshort),
            "skill_coverage": (tot_sreq - tot_sshort) / tot_sreq if tot_sreq else 0.0,
            "workload_gini": float(gini), "overtime": overtime,
            "preference_score": float(pref_mean),
        }
        self._ctx_cache = {
            "date": date, "groups": groups, "assign": assign, "wk": wk, "order": order,
            "w_vec": w_vec, "contracts": contracts, "idx": {s: i for i, s in enumerate(order)},
            "gini": float(gini), "overtime": overtime, "pref_sum": float(pref_sum),
            "n_today": n_today, "pref_mean": float(pref_mean), "prev": prev, "next": nxt,
            "metrics": metrics, "free": {},
        }
        return self._ctx_cache

    def _free_pool(self, ctx, shift: str) -> List[str]:
        """Staff who could be called in for `shift` today (computed once per shift per state)."""
        pool = ctx["free"].get(shift)
        if pool is None:
            date, hours = ctx["date"], self.cfg.shift_hours.get(shift, 8)
            pool = []
            for sid in self.staff_hours:
                if sid in ctx["assign"] or sid in self.absent_staff:
                    continue
                if not self._availability(sid, date, shift) or self._request_off(sid, date, shift):
                    continue
                if not self._rest_ok(ctx, sid, shift):
                    continue
                if ctx["wk"].get(sid, 0.0) + hours > self.staff_hours[sid] + self.cfg.overtime_allowance + 1e-9:
                    continue
                pool.append(sid)
            ctx["free"][shift] = pool
        return pool

    # --------------------------------------------------------------- metrics
    def _all_metrics(self) -> Dict[str, float]:
        return dict(self._ctx()["metrics"])

    def _potential(self, m: Dict[str, float]) -> float:
        c = self.cfg
        return -(c.w_short * m["shortage"] + c.w_skill * m["skill_shortage"]
                 + c.w_overtime * m["overtime"] / 8.0 + c.w_gini * m["workload_gini"]
                 + c.w_pref * (1.0 - m["preference_score"]))

    # ----------------------------------------------------------- feasibility
    def _rest_ok(self, ctx, sid: str, shift: str) -> bool:
        for p in ctx["prev"].get(sid, _EMPTY):
            if (p, shift) in self._forb:
                return False
        for n in ctx["next"].get(sid, _EMPTY):
            if (shift, n) in self._forb:
                return False
        return True

    def _skill_gate(self, ctx, sid: str, tgt) -> bool:
        g = ctx["groups"][tgt]
        if not g["needed"] or g["short"] > 0:   # an extra body always helps head-count
            return True
        sk = self.skill_map.get(sid, _EMPTY)
        return any(c in sk for c in g["needed"])

    def _move_ok(self, ctx, sid: str, src, tgt) -> bool:
        if src[1] != tgt[1] or src == tgt:
            return False
        if ctx["assign"].get(sid) != src or tgt not in ctx["groups"]:
            return False
        if sid in self.absent_staff:
            return False
        date, shift = ctx["date"], tgt[1]
        if not self._availability(sid, date, shift) or self._request_off(sid, date, shift):
            return False
        if not self._rest_ok(ctx, sid, shift):
            return False
        if ctx["wk"].get(sid, 0.0) > self.staff_hours.get(sid, 40.0):
            return False
        return self._skill_gate(ctx, sid, tgt)

    def _callin_ok(self, ctx, sid: str, tgt) -> bool:
        if sid in ctx["assign"] or sid in self.absent_staff or tgt not in ctx["groups"]:
            return False
        date, shift = ctx["date"], tgt[1]
        if not self._availability(sid, date, shift) or self._request_off(sid, date, shift):
            return False
        if not self._rest_ok(ctx, sid, shift):
            return False
        hours = self.cfg.shift_hours.get(shift, 8)
        if ctx["wk"].get(sid, 0.0) + hours > self.staff_hours.get(sid, 40.0) + self.cfg.overtime_allowance + 1e-9:
            return False
        return self._skill_gate(ctx, sid, tgt)

    def _is_valid_move(self, sid, source_unit, source_shift, target_unit, target_shift):
        return self._move_ok(self._ctx(), str(sid),
                             (str(source_unit), normalize_shift(source_shift)),
                             (str(target_unit), normalize_shift(target_shift)))

    # ------------------------------------------------------ candidate actions
    def get_valid_actions(self) -> List[Dict]:
        acts: List[Dict] = [{"type": "no_op"}]
        entries: List[Dict] = []
        if (not self._suspend and not self.done and self.current_date is not None
                and self.moves_today < self.cfg.moves_per_day):
            entries = self._generate_entries()
        entries.sort(key=lambda e: -e["est"])
        entries = entries[: self.cfg.max_candidates]

        left = 1.0 - self.moves_today / max(1, self.cfg.moves_per_day)
        self.cand_feats[:] = 0.0
        self.cand_mask[:] = False
        self.cand_mask[0] = True
        self.cand_feats[0, 0] = 1.0
        self.cand_feats[0, 14] = left
        self.teacher_idx = 0
        for i, e in enumerate(entries, start=1):
            e["feat"][14] = left
            self.cand_feats[i] = e["feat"]
            self.cand_mask[i] = True
            if self.teacher_idx == 0 and e["est"] > 0 and e["action"]["gain_short"] >= 0 and e["action"]["gain_skill"] >= 0:
                self.teacher_idx = i
            acts.append(e["action"])
        return acts

    def _generate_entries(self) -> List[Dict]:
        c = self.cfg
        ctx = self._ctx()
        groups, assign = ctx["groups"], ctx["assign"]
        targets = [k for k, g in groups.items() if g["short"] > 0 or g["ss"] > 0]
        if not targets:
            return []
        evt = {(str(i.get("unit", "")), normalize_shift(str(i.get("shift", "Day"))))
               for i in (self.last_event or {}).get("demand_surge", [])}
        targets.sort(key=lambda k: (k in evt, 10 * groups[k]["short"] + groups[k]["ss"]), reverse=True)
        targets = targets[: self.cfg.max_targets]
        date = ctx["date"]
        date_s = str(date.date())
        event_targets = {(str(i.get("unit", "")), normalize_shift(str(i.get("shift", "Day"))))
                         for i in (self.last_event or {}).get("demand_surge", [])}
        entries: List[Dict] = []
        sm = self.skill_map

        for tgt in targets:
            tg = groups[tgt]
            tshort, tss = tg["short"], tg["ss"]
            tgt_short_f = tshort / max(1, tg["req"])
            tgt_ss_f = tss / max(1, sum(tg["skills"].values()))
            is_evt = 1.0 if tgt in event_targets else 0.0

            # ---- reassign (V3 action) -------------------------------------
            for sid, src in assign.items():
                if src[1] != tgt[1] or src == tgt or src not in groups:
                    continue
                if not self._move_ok(ctx, sid, src, tgt):
                    continue
                sg = groups[src]
                ns = self._gap([x for x in sg["ids"] if x != sid], sg["req"], sg["skills"], sm)
                nt = self._gap(tg["ids"] + [sid], tg["req"], tg["skills"], sm)
                gs = (sg["short"] + tshort) - (ns[0] + nt[0])
                gk = (sg["ss"] + tss) - (ns[1] + nt[1])
                est = c.w_short * gs + c.w_skill * gk - c.cost_reassign
                surplus_after = (len(sg["ids"]) - 1 - sg["req"]) / max(1, sg["req"])
                feat = np.zeros(FEAT_DIM, dtype=np.float32)
                feat[1] = 1.0; feat[3] = gs; feat[4] = gk; feat[8] = est
                feat[9] = tgt_short_f; feat[10] = tgt_ss_f; feat[11] = surplus_after
                feat[12] = ctx["wk"].get(sid, 0.0) / max(1.0, self.staff_hours.get(sid, 40.0))
                feat[13] = is_evt
                entries.append({"est": est, "feat": feat, "action": {
                    "type": "reassign", "staff_id": sid, "date": date_s,
                    "source_unit": src[0], "source_shift": src[1],
                    "target_unit": tgt[0], "target_shift": tgt[1],
                    "gain_short": float(gs), "gain_skill": float(gk)}})

            # ---- call-in (new action) -------------------------------------
            if not c.allow_call_in:
                continue
            hours = c.shift_hours.get(tgt[1], 8)
            needed = tg["needed"]
            pool = []
            for sid in self._free_pool(ctx, tgt[1]):
                if not self._skill_gate(ctx, sid, tgt):
                    continue
                match = sum(code in sm.get(sid, _EMPTY) for code in needed)
                wk_h = ctx["wk"].get(sid, 0.0)
                pool.append((-match, wk_h / max(1.0, self.staff_hours[sid]),
                             -self._preference(sid, date, tgt[1]), sid))
            pool.sort()
            for _, ratio, negpref, sid in pool[: c.call_in_per_target]:
                nt = self._gap(tg["ids"] + [sid], tg["req"], tg["skills"], sm)
                gs = tshort - nt[0]
                gk = tss - nt[1]
                wv = ctx["w_vec"].copy()
                i = ctx["idx"][sid]
                old_h = wv[i]
                wv[i] += hours
                d_gini = compute_gini(wv) - ctx["gini"]
                cap = self.staff_hours[sid]
                d_ot = max(0.0, wv[i] - cap) - max(0.0, old_h - cap)
                p_new = -negpref
                new_mean = (ctx["pref_sum"] + p_new) / (ctx["n_today"] + 1)
                d_pref = new_mean - ctx["pref_mean"]
                est = (c.w_short * gs + c.w_skill * gk - c.w_gini * d_gini
                       - c.w_overtime * d_ot / 8.0 + c.w_pref * d_pref - c.cost_call_in)
                feat = np.zeros(FEAT_DIM, dtype=np.float32)
                feat[2] = 1.0; feat[3] = gs; feat[4] = gk; feat[5] = d_gini * 10
                feat[6] = d_ot / 8.0; feat[7] = d_pref * 10; feat[8] = est
                feat[9] = tgt_short_f; feat[10] = tgt_ss_f
                feat[12] = ratio; feat[13] = is_evt
                entries.append({"est": est, "feat": feat, "action": {
                    "type": "call_in", "staff_id": sid, "date": date_s,
                    "target_unit": tgt[0], "target_shift": tgt[1],
                    "gain_short": float(gs), "gain_skill": float(gk)}})
        return entries

    # ---------------------------------------------------------------- actions
    def _apply_action(self, action: Dict) -> bool:
        t = action.get("type")
        if t == "no_op":
            return True
        if t == "reassign":
            ok = super()._apply_action(action)
            self._ctx_cache = None
            if ok:
                self.n_changes += 1
            return ok
        if t == "call_in":
            ctx = self._ctx()
            sid = str(action["staff_id"])
            tgt = (str(action["target_unit"]), normalize_shift(str(action["target_shift"])))
            if not self._callin_ok(ctx, sid, tgt):
                return False
            date = ctx["date"]
            row = {"staff_id": sid, "date": date, "unit": tgt[0], "shift": tgt[1], "assigned": 1,
                   "preference_score": self._preference(sid, date, tgt[1]),
                   "shift_hours": float(self.cfg.shift_hours.get(tgt[1], 8))}
            self.roster = pd.concat([self.roster, pd.DataFrame([row])], ignore_index=True)
            self._ctx_cache = None
            self.n_changes += 1
            return True
        return False

    def repair_day(self, time_limit: float = 5.0, change_cost: int = 3) -> int:
        """Exact minimal-change CP-SAT re-optimisation of *today only*.

        Hard: one shift/day, availability, request-off, rest, weekly contract
        hours (other days fixed), absent staff excluded.  Objective: uncovered
        and skill-short slack first, then number of changed staff, fairness and
        preference.  Returns the number of staff whose assignment changed.
        """
        from ortools.sat.python import cp_model

        ctx = self._ctx()
        groups, cur, wk = ctx["groups"], ctx["assign"], ctx["wk"]
        if not groups:
            return 0
        date = ctx["date"]
        keys = list(groups)
        hrs = lambda sh: self.cfg.shift_hours.get(sh, 8)
        model = cp_model.CpModel()
        x: Dict[Tuple[str, Tuple[str, str]], cp_model.IntVar] = {}
        wk_other: Dict[str, float] = {}
        for sid in self.staff_hours:
            if sid in self.absent_staff or (sid in cur and cur[sid] not in groups):
                continue
            w_o = wk.get(sid, 0.0) - (hrs(cur[sid][1]) if sid in cur else 0.0)
            wk_other[sid] = w_o
            for key in keys:
                if cur.get(sid) == key:
                    ok = True
                else:
                    ok = (self._availability(sid, date, key[1]) and not self._request_off(sid, date, key[1])
                          and self._rest_ok(ctx, sid, key[1])
                          and w_o + hrs(key[1]) <= self.staff_hours[sid] + self.cfg.overtime_allowance + 1e-9)
                if ok:
                    x[(sid, key)] = model.NewBoolVar(f"x_{sid}_{key[0]}_{key[1]}")
        by_sid: Dict[str, List] = {}
        for (sid, key), v in x.items():
            by_sid.setdefault(sid, []).append(v)
        for sid, vs in by_sid.items():
            model.Add(sum(vs) <= 1)

        obj = []
        for key in keys:
            g = groups[key]
            members = [(sid, v) for (sid, k), v in x.items() if k == key]
            unc = model.NewIntVar(0, g["req"], f"unc_{key}")
            model.Add(sum(v for _, v in members) + unc >= g["req"])
            obj.append(1000 * unc)
            for code, n in g["skills"].items():
                sk = model.NewIntVar(0, n, f"sk_{key}_{code}")
                qual = [v for sid, v in members if code in self.skill_map.get(sid, _EMPTY)]
                model.Add(sum(qual) + sk >= n)
                obj.append(900 * sk)
        for (sid, key), v in x.items():
            pen = int(round((1.0 - self._preference(sid, date, key[1])) * 10))
            fair = int(round(4 * wk_other[sid] / max(1.0, self.staff_hours[sid])))
            obj.append((pen + fair) * v)
        for sid in by_sid:
            if sid in cur:
                obj.append(change_cost * (1 - x[(sid, cur[sid])]))
            else:
                obj.append(change_cost * sum(by_sid[sid]))
        model.Minimize(sum(obj))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit)
        solver.parameters.num_search_workers = 2
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return 0

        new = {sid: key for (sid, key), v in x.items() if solver.Value(v)}
        changes, remove, add = 0, set(), []
        for sid in by_sid:
            old = cur.get(sid)
            nk = new.get(sid)
            if old != nk:
                changes += 1
                if old is not None:
                    remove.add(sid)
                if nk is not None:
                    add.append((sid, nk))
        if remove:
            m = self.roster["staff_id"].isin(remove) & (self.roster["date"] == date)
            self.roster = self.roster[~m].reset_index(drop=True)
        if add:
            rows = [{"staff_id": s, "date": date, "unit": k[0], "shift": k[1], "assigned": 1,
                     "preference_score": self._preference(s, date, k[1]),
                     "shift_hours": float(hrs(k[1]))} for s, k in add]
            self.roster = pd.concat([self.roster, pd.DataFrame(rows)], ignore_index=True)
        self._ctx_cache = None
        self.n_changes += changes
        return changes

    # ------------------------------------------------------------------ state
    def get_state(self) -> np.ndarray:
        ctx = self._ctx()
        m = ctx["metrics"]
        groups = ctx["groups"]
        n_groups = max(1, len(groups))
        base = [
            m["coverage"], m["shortage"] / max(1.0, m["required"]), m["skill_coverage"],
            m["skill_shortage"] / max(1.0, m["skill_required"]), m["workload_gini"],
            m["overtime"] / max(1.0, 8.0 * len(self.staff)), m["preference_score"],
            len(self.absent_staff) / max(1, len(self.staff)),
            len((self.last_event or {}).get("demand_surge", [])) / n_groups,
            self.current_index / max(1, len(self.episode_dates) - 1),
            self.moves_today / max(1, self.cfg.moves_per_day),
        ]
        comps: List[float] = []
        for unit in self._units:
            for shift in self.SHIFTS:
                g = groups.get((unit, shift))
                if g is None or g["req"] == 0:
                    comps += [0.0, 0.0, 0.0]
                else:
                    comps += [len(g["ids"]) / g["req"], g["short"] / g["req"],
                              g["ss"] / max(1, sum(g["skills"].values()))]
        return np.asarray(base + comps, dtype=np.float32)

    # ---------------------------------------------------------------- stepping
    def reset(self, start_date=None, forced_events=None):
        self.moves_today = 0
        self.n_changes = 0
        self._ctx_cache = None
        self._suspend = True
        try:
            super().reset(start_date, forced_events)
        finally:
            self._suspend = False
        self._ctx_cache = None
        self.day_open = self._all_metrics()
        self.candidates = self.get_valid_actions()
        return self.get_state()

    def _advance_day(self):
        self.current_index += 1
        self.moves_today = 0
        if self.current_index >= len(self.episode_dates):
            self.done = True
            self.candidates = self.get_valid_actions()   # -> [no_op]
            return
        self.current_date = self.episode_dates[self.current_index]
        self.current_shift = "Day"
        self.last_event = self._inject_event_for_current_date()
        self._remove_absent_assignments()
        self.day_open = self._all_metrics()
        self.candidates = self.get_valid_actions()

    def step(self, action_index: int):
        if self.done:
            raise RuntimeError("Episode is already done. Call reset().")
        before = self._all_metrics()
        idx = int(action_index)
        valid_index = 0 <= idx < len(self.candidates)
        action = self.candidates[idx] if valid_index else {"type": "invalid"}
        changed, valid = False, valid_index
        if valid_index:
            valid = self._apply_action(action)
            changed = bool(valid and action["type"] in ("reassign", "call_in"))
        after = self._all_metrics()

        reward = self._potential(after) - self._potential(before)
        if changed:
            reward -= self.cfg.cost_call_in if action["type"] == "call_in" else self.cfg.cost_reassign
        if not valid:
            reward -= self.cfg.invalid_penalty

        info = {"date": str(pd.Timestamp(self.current_date).date()), "action": action,
                "valid_action": bool(valid), "changed": changed, "before": before, "after": after,
                "event": dict(self.last_event), "day_closed": False}
        if changed:
            self.moves_today += 1
            self.candidates = self.get_valid_actions()
        else:                                    # no-op / invalid -> close the day
            info.update(day_closed=True, day_open=dict(self.day_open), day_close=dict(after))
            self._advance_day()
        info["candidate_count"] = len(self.candidates)
        return self.get_state(), float(reward), self.done, info


# =============================================================================
# Policy network, normaliser, PPO
# =============================================================================
class RunningNorm:
    def __init__(self, dim: int):
        self.n, self.mean, self.var = 1e-4, np.zeros(dim), np.ones(dim)

    def update(self, x: np.ndarray):
        x = np.atleast_2d(x)
        bn, bm, bv = x.shape[0], x.mean(0), x.var(0)
        d, tot = bm - self.mean, self.n + bn
        self.mean = self.mean + d * bn / tot
        self.var = (self.var * self.n + bv * bn + d ** 2 * self.n * bn / tot) / tot
        self.n = tot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -5, 5).astype(np.float32)


class CandidateScorer(nn.Module):
    """Index-free policy: one shared network scores every candidate."""

    def __init__(self, state_dim: int, cand_dim: int, h: int = 128):
        super().__init__()
        self.state_enc = nn.Sequential(nn.Linear(state_dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.cand_enc = nn.Sequential(nn.Linear(cand_dim, h), nn.Tanh())
        self.scorer = nn.Sequential(nn.Linear(3 * h, h), nn.Tanh(), nn.Linear(h, 1))
        self.value = nn.Linear(h, 1)

    def forward(self, s, c, mask):
        hs = self.state_enc(s)
        hc = self.cand_enc(c)
        hs_e = hs.unsqueeze(1).expand(-1, hc.shape[1], -1)
        logits = self.scorer(torch.cat([hs_e, hc, hs_e * hc], -1)).squeeze(-1)
        return logits.masked_fill(~mask, -1e9), self.value(hs).squeeze(-1)


class PPOAgent:
    def __init__(self, state_dim: int, cfg: V45PPOConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = CandidateScorer(state_dim, FEAT_DIM, cfg.hidden_size).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

    @torch.no_grad()
    def act(self, s, c, mask, deterministic=False):
        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=self.device).unsqueeze(0)
        logits, v = self.model(t(s, torch.float32), t(c, torch.float32), t(mask, torch.bool))
        dist = Categorical(logits=logits)
        a = torch.argmax(logits, -1) if deterministic else dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(v.item())

    def _tensors(self, b):
        d = self.device
        return (torch.as_tensor(b["states"], dtype=torch.float32, device=d),
                torch.as_tensor(b["cands"], dtype=torch.float32, device=d),
                torch.as_tensor(b["masks"], dtype=torch.bool, device=d),
                torch.as_tensor(b["actions"], dtype=torch.long, device=d),
                torch.as_tensor(b["logp"], dtype=torch.float32, device=d),
                torch.as_tensor(b["returns"], dtype=torch.float32, device=d),
                torch.as_tensor(b["adv"], dtype=torch.float32, device=d),
                torch.as_tensor(b["teacher"], dtype=torch.long, device=d))

    def update_bc(self, b, epochs: int):
        S, C, M, A, LP, R, ADV, T = self._tensors(b)
        n, losses = len(S), []
        for _ in range(epochs):
            perm = torch.randperm(n)
            for st in range(0, n, self.cfg.minibatch_size):
                i = perm[st: st + self.cfg.minibatch_size]
                logits, v = self.model(S[i], C[i], M[i])
                loss = nn.functional.cross_entropy(logits, T[i]) + \
                    self.cfg.value_coef * nn.functional.smooth_l1_loss(v, R[i])
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step(); losses.append(float(loss.item()))
        return float(np.mean(losses))

    def update(self, b, bc_coef: float):
        S, C, M, A, LP, R, ADV, T = self._tensors(b)
        ADV = (ADV - ADV.mean()) / (ADV.std(unbiased=False) + 1e-8)
        n, losses, kls = len(S), [], []
        for _ in range(self.cfg.update_epochs):
            perm = torch.randperm(n)
            for st in range(0, n, self.cfg.minibatch_size):
                i = perm[st: st + self.cfg.minibatch_size]
                logits, v = self.model(S[i], C[i], M[i])
                dist = Categorical(logits=logits)
                lp = dist.log_prob(A[i])
                ratio = torch.exp(lp - LP[i])
                s1 = ratio * ADV[i]
                s2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * ADV[i]
                pol = -torch.min(s1, s2).mean()
                vf = nn.functional.smooth_l1_loss(v, R[i])
                bc = nn.functional.cross_entropy(logits, T[i])
                loss = pol + self.cfg.value_coef * vf - self.cfg.entropy_coef * dist.entropy().mean() + bc_coef * bc
                if not torch.isfinite(loss):
                    continue
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                losses.append(float(loss.item()))
                kls.append(float((LP[i] - lp).mean().item()))
            if kls and np.mean(kls[-4:]) > self.cfg.target_kl:
                break
        return float(np.mean(losses)) if losses else 0.0

    def save(self, path, norm: RunningNorm):
        torch.save({"model": self.model.state_dict(), "cfg": asdict(self.cfg),
                    "norm": {"n": norm.n, "mean": norm.mean, "var": norm.var}}, path)


# =============================================================================
# Scenario generation
# =============================================================================
def _roster_index(roster: pd.DataFrame):
    r = roster.copy()
    r["date"] = pd.to_datetime(r["date"])
    return {d: g["staff_id"].astype(str).unique().tolist() for d, g in r.groupby("date")}


def make_random_events(staff_by_date, demand: pd.DataFrame, dates, rng: random.Random,
                       surge_prob=0.5, max_groups=2, max_abs=4, extra=(1, 4)):
    events = []
    for d in dates:
        rows = demand[demand["date"] == d]
        surge = []
        if len(rows) and rng.random() < surge_prob:
            for i in rng.sample(range(len(rows)), min(rng.randint(1, max_groups), len(rows))):
                rr = rows.iloc[i]
                surge.append({"unit": str(rr["unit"]), "shift": normalize_shift(rr["shift"]),
                              "extra_staff": rng.randint(*extra)})
        ds = staff_by_date.get(pd.Timestamp(d), [])
        absent = rng.sample(ds, rng.randint(0, min(max_abs, len(ds)))) if ds else []
        events.append({"date": str(pd.Timestamp(d).date()), "demand_surge": surge, "absent_staff": absent})
    return events


def make_v44_stress_events(staff_by_date, demand: pd.DataFrame, all_dates, dates):
    """The exact V4.4 forced scenario: +2 surge on a rotating target and 2 absences on day 1."""
    start = dates[0]
    rows = demand[demand["date"] == start].sort_values(["required_staff", "unit", "shift"],
                                                         ascending=[False, True, True])
    target = rows.iloc[all_dates.index(start) % len(rows)]
    events = [{"date": str(start.date()),
               "demand_surge": [{"unit": str(target["unit"]), "shift": normalize_shift(target["shift"]),
                                 "extra_staff": 2}],
               "absent_staff": staff_by_date.get(start, [])[:2]}]
    for d in dates[1:]:
        events.append({"date": str(pd.Timestamp(d).date()), "demand_surge": [], "absent_staff": []})
    return events


# =============================================================================
# Rollouts and training
# =============================================================================
def teacher_policy(env, state=None, repaired=False):
    return env.teacher_idx


def collect_episode(env, agent, norm, events, start, mode="ppo", explore=0.0, cfg=None, rng=None):
    state = env.reset(start_date=start, forced_events=events)
    S, C, M, A, LP, V, R, T = [], [], [], [], [], [], [], []
    while not env.done:
        norm.update(state)
        s = norm(state)
        c, m = env.cand_feats.copy(), env.cand_mask.copy()
        a, lp, v = agent.act(s, c, m)
        teacher = env.teacher_idx
        if mode == "teacher":
            a = teacher
            if rng.random() < explore:
                a = rng.choice(np.flatnonzero(m).tolist())
        state, r, done, _ = env.step(a)
        S.append(s); C.append(c); M.append(m); A.append(a); LP.append(lp); V.append(v); R.append(r); T.append(teacher)
    T_ = len(R)
    adv = np.zeros(T_, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T_)):
        nv = V[t + 1] if t < T_ - 1 else 0.0
        delta = R[t] + cfg.gamma * nv - V[t]
        gae = delta + cfg.gamma * cfg.gae_lambda * gae
        adv[t] = gae
    ret = adv + np.asarray(V, dtype=np.float32)
    return {"states": np.asarray(S, np.float32), "cands": np.asarray(C, np.float32),
            "masks": np.asarray(M, bool), "actions": np.asarray(A, np.int64),
            "logp": np.asarray(LP, np.float32), "returns": ret, "adv": adv,
            "teacher": np.asarray(T, np.int64)}, float(np.sum(R))


def _merge(batches):
    return {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}


def train_agent(env, data, roster, cfg: V45PPOConfig, device="cpu"):
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    demand = env.demand
    all_dates = sorted(pd.to_datetime(demand["date"].unique()))
    n_days = env.cfg.episode_days
    starts = all_dates[: max(1, len(all_dates) - n_days + 1)]
    staff_by_date = _roster_index(roster)
    state_dim = len(env.reset(start_date=str(starts[0].date())))
    agent = PPOAgent(state_dim, cfg, device)
    norm = RunningNorm(state_dim)
    history = []

    def sample_events(frac):
        start = rng.choice(starts)
        dates = [d for d in all_dates if d >= start][:n_days]
        if rng.random() < 0.25:
            ev = make_v44_stress_events(staff_by_date, demand, all_dates, dates)
        else:
            ev = make_random_events(staff_by_date, demand, dates, rng,
                                    surge_prob=rng.uniform(0.2, 0.7),
                                    max_abs=int(round(2 + 3 * frac)))
        return str(start.date()), ev

    t_start = time.perf_counter()
    for u in range(cfg.bc_updates):                       # behaviour-cloning warm-up
        bs = []
        for e in range(cfg.episodes_per_update):
            st, ev = sample_events(0.3)
            b, _ = collect_episode(env, agent, norm, ev, st, "teacher", 0.15, cfg, rng)
            bs.append(b)
            logger.info("  BC %d/%d episode %d/%d | %d steps | %.0fs elapsed", u + 1, cfg.bc_updates,
                        e + 1, cfg.episodes_per_update, len(b["actions"]), time.perf_counter() - t_start)
        loss = agent.update_bc(_merge(bs), cfg.bc_epochs)
        logger.info("BC warm-up %d/%d | loss=%.4f", u + 1, cfg.bc_updates, loss)

    for u in range(cfg.updates):
        frac = u / max(1, cfg.updates - 1)
        bc_coef = cfg.bc_coef_start * (1 - frac) + cfg.bc_coef_end * frac
        bs, rets = [], []
        for e in range(cfg.episodes_per_update):
            st, ev = sample_events(frac)
            b, ret = collect_episode(env, agent, norm, ev, st, "ppo", 0.0, cfg, rng)
            bs.append(b); rets.append(ret)
            logger.info("  PPO %d/%d episode %d/%d | %d steps | %.0fs elapsed", u + 1, cfg.updates,
                        e + 1, cfg.episodes_per_update, len(b["actions"]), time.perf_counter() - t_start)
        loss = agent.update(_merge(bs), bc_coef)
        history.append({"update": u + 1, "mean_return": float(np.mean(rets)), "loss": loss, "bc_coef": bc_coef})
        if u == 0 or (u + 1) % 5 == 0:
            logger.info("PPO update %d/%d | return=%.3f | loss=%.4f", u + 1, cfg.updates, np.mean(rets), loss)
    return agent, norm, history


# =============================================================================
# Policies and evaluation
# =============================================================================
def _safe(env, idx: int) -> bool:
    a = env.candidates[idx]
    return a.get("gain_short", 0.0) >= 0 and a.get("gain_skill", 0.0) >= 0


def make_ppo_policy(agent, norm, repair_fallback=False, gate=True):
    def policy(env, state, repaired):
        idx, _, _ = agent.act(norm(state), env.cand_feats, env.cand_mask, deterministic=True)
        if gate and idx > 0 and not _safe(env, idx):
            idx = 0
        if idx == 0 and repair_fallback and not repaired:
            m = env._all_metrics()
            if m["shortage"] + m["skill_shortage"] > 0:
                return REPAIR
        return idx
    return policy


def noop_policy(env, state, repaired):
    return 0


def greedy_policy(env, state, repaired):
    return env.teacher_idx


def cpsat_policy(env, state, repaired):
    return 0 if repaired else REPAIR


def _summarise(opens, closes, n_changes, solve_t, total_reward=0.0):
    keys = ["coverage", "shortage", "skill_shortage", "skill_coverage", "workload_gini", "overtime",
            "preference_score"]
    out = {k: float(np.mean([c[k] for c in closes])) for k in keys}
    o_s = sum(o["shortage"] + o["skill_shortage"] for o in opens)
    c_s = sum(c["shortage"] + c["skill_shortage"] for c in closes)
    out["open_shortage"] = float(np.mean([o["shortage"] for o in opens]))
    out["recovery"] = float(1 - c_s / o_s) if o_s > 0 else float("nan")
    out["days_fully_covered"] = float(np.mean([c["shortage"] == 0 and c["skill_shortage"] == 0 for c in closes]))
    out["changes"] = float(n_changes)
    out["solve_seconds"] = float(solve_t)
    out["return"] = float(total_reward)
    return out


def run_episode(env, policy, events, start, repair_time=5.0):
    state = env.reset(start_date=start, forced_events=events)
    opens, closes, repaired, t_rep, tot = [], [], False, 0.0, 0.0
    while not env.done:
        idx = policy(env, state, repaired)
        if idx == REPAIR:
            t0 = time.perf_counter()
            env.repair_day(repair_time)
            t_rep += time.perf_counter() - t0
            repaired, idx = True, 0
        state, r, done, info = env.step(idx)
        tot += r
        if info["day_closed"]:
            opens.append(info["day_open"]); closes.append(info["day_close"]); repaired = False
    return _summarise(opens, closes, env.n_changes, t_rep, tot)


def run_v44_episode(v44, agent44, data, roster, events, start, days, seed):
    cfg44 = v44.V4PPOConfig(episode_days=days)
    env = v44.make_env(data, roster, cfg44, seed)
    state = env.reset(start_date=start, forced_events=events)
    opens, closes, ch = [], [], 0
    while not env.done:
        idx = v44.select_adaptive_action(env, agent44, state)
        state, _, _, info = env.step(idx)
        opens.append(info["before"]); closes.append(info["after"]); ch += int(info["changed"])
    return _summarise(opens, closes, ch, 0.0)


def _bootstrap(d: np.ndarray, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def evaluate(env, data, roster, agent, norm, cfg_env, episodes, seed, include_v44, repair_time):
    demand = env.demand
    all_dates = sorted(pd.to_datetime(demand["date"].unique()))
    n_days = cfg_env.episode_days
    starts = all_dates[: max(1, len(all_dates) - n_days + 1)]
    staff_by_date = _roster_index(roster)
    policies = {
        "noop": noop_policy,
        "greedy_multi": greedy_policy,
        "ppo": make_ppo_policy(agent, norm),
        "ppo+repair": make_ppo_policy(agent, norm, repair_fallback=True),
        "cpsat_repair": cpsat_policy,
    }
    v44 = agent44 = None
    if include_v44:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import hospital_hrm_framework_v4_4 as v44  # type: ignore
        except Exception as exc:
            logger.warning("V4.4 comparison disabled (%s)", exc)
            v44 = None
    rows = []
    for suite in ("v44_stress", "random_disruption"):
        for i in range(episodes):
            start = starts[i % len(starts)]
            dates = [d for d in all_dates if d >= start][:n_days]
            rng = random.Random(seed + 1000 * (suite == "random_disruption") + i)
            events = (make_v44_stress_events(staff_by_date, demand, all_dates, dates)
                      if suite == "v44_stress"
                      else make_random_events(staff_by_date, demand, dates, rng))
            st = str(start.date())
            for name, pol in policies.items():
                env.cfg.seed = seed + i
                res = run_episode(env, pol, events, st, repair_time)
                rows.append({"suite": suite, "episode": i, "method": name, **res})
            if v44 is not None:
                if agent44 is None:
                    s0 = v44.make_env(data, roster, v44.V4PPOConfig(episode_days=n_days), seed).reset(start_date=st)
                    agent44 = v44.PPOAgent(len(s0), v44.V4PPOConfig().max_candidates + 1, v44.V4PPOConfig())
                res = run_v44_episode(v44, agent44, data, roster, events, st, n_days, seed + i)
                rows.append({"suite": suite, "episode": i, "method": "v44_lookahead", **res})
        logger.info("evaluated suite %s", suite)
    return pd.DataFrame(rows)


def summarise_eval(df: pd.DataFrame) -> Tuple[Dict, str]:
    metrics = ["coverage", "shortage", "skill_shortage", "workload_gini", "overtime",
               "preference_score", "recovery", "days_fully_covered", "changes", "solve_seconds"]
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    summary, lines = {}, []
    for suite, g in df.groupby("suite"):
        means = g.groupby("method")[metrics].mean()
        summary[suite] = {"means": means.round(4).to_dict("index"), "paired_vs": {}}
        lines += [f"\n### Suite: {suite}  (end-of-day metrics averaged over days; n={g['episode'].nunique()} paired episodes)\n",
                  means.round(3).to_markdown(), ""]
        for ref in ("noop", "v44_lookahead"):
            if ref not in means.index:
                continue
            rows = []
            for meth in means.index:
                if meth == ref:
                    continue
                a = g[g.method == meth].sort_values("episode")
                b = g[g.method == ref].sort_values("episode")
                rec = {"method": meth}
                for k in ("shortage", "skill_shortage", "workload_gini"):
                    d = (a[k].values - b[k].values)
                    lo, hi = _bootstrap(d)
                    p = float("nan")
                    if wilcoxon is not None and np.any(d != 0):
                        try:
                            p = float(wilcoxon(d).pvalue)
                        except Exception:
                            pass
                    rec[f"d_{k}"] = f"{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3g}"
                rows.append(rec)
            summary[suite]["paired_vs"][ref] = rows
            lines += [f"Paired difference vs **{ref}** (negative = better; mean [95% bootstrap CI], Wilcoxon p):\n",
                      pd.DataFrame(rows).to_markdown(index=False), ""]
    return summary, "\n".join(lines)


# =============================================================================
# Runner
# =============================================================================
def run_v4_5(dataset_path, roster_path=None, output_dir="v4_5_outputs", updates=60,
             episodes_per_update=8, eval_episodes=40, time_limit=120, include_v44=True,
             repair_time=5.0, seed=42, device="cpu", evaluate_only=False, model_path=None):
    os.makedirs(output_dir, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    data, roster = load_v2_baseline(dataset_path, roster_path, time_limit)
    cfg_env = V45Config(seed=seed)
    env = AdaptiveWorkforceEnv(data, roster, cfg_env)

    if evaluate_only:
        checkpoint_path = model_path or os.path.join(output_dir, "v4_5_model.pt")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"V4.5 checkpoint not found: {checkpoint_path}. "
                "Pass the trained checkpoint using --model."
            )
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dim = len(env.reset())
        saved_cfg = checkpoint.get("cfg", {})
        cfg = V45PPOConfig(**{k: v for k, v in saved_cfg.items()
                              if k in V45PPOConfig.__dataclass_fields__})
        agent = PPOAgent(state_dim, cfg, device)
        agent.model.load_state_dict(checkpoint["model"])
        norm_data = checkpoint.get("norm", {})
        norm = RunningNorm(state_dim)
        norm.n = float(norm_data.get("n", 1e-4))
        norm.mean = np.asarray(norm_data.get("mean", np.zeros(state_dim)), dtype=float)
        norm.var = np.asarray(norm_data.get("var", np.ones(state_dim)), dtype=float)
        logger.info("Loaded trained V4.5 checkpoint: %s", checkpoint_path)
        train_s = None
    else:
        cfg = V45PPOConfig(seed=seed, updates=updates, episodes_per_update=episodes_per_update)
        t0 = time.perf_counter()
        agent, norm, history = train_agent(env, data, roster, cfg, device)
        train_s = time.perf_counter() - t0
        agent.save(os.path.join(output_dir, "v4_5_model.pt"), norm)
        with open(os.path.join(output_dir, "v4_5_training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    df = evaluate(env, data, roster, agent, norm, cfg_env, eval_episodes, seed, include_v44, repair_time)
    df.to_csv(os.path.join(output_dir, "v4_5_eval_episodes.csv"), index=False)
    summary, md = summarise_eval(df)
    if train_s is not None:
        summary["train_seconds"] = train_s
    summary["config"] = {"env": asdict(cfg_env), "ppo": asdict(cfg)}
    with open(os.path.join(output_dir, "v4_5_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(output_dir, "v4_5_report.md"), "w") as f:
        f.write("# V4.5 evaluation\\n" + md)
    print(md)
    return output_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train/evaluate Hospital HRM V4.5")
    p.add_argument("--dataset", required=True)
    p.add_argument("--roster", default=None)
    p.add_argument("--output", default="v4_5_outputs")
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--episodes-per-update", type=int, default=8)
    p.add_argument("--eval-episodes", type=int, default=40)
    p.add_argument("--time-limit", type=int, default=120)
    p.add_argument("--repair-time", type=float, default=5.0)
    p.add_argument("--skip-v44", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--evaluate-only", action="store_true",
                   help="Evaluate an existing V4.5 checkpoint without training")
    p.add_argument("--model", default=None,
                   help="Path to a checkpoint created by hospital_hrm_framework_v4_5.py")
    a = p.parse_args()
    run_v4_5(a.dataset, a.roster, a.output, a.updates, a.episodes_per_update, a.eval_episodes,
             a.time_limit, not a.skip_v44, a.repair_time, a.seed, a.device,
             a.evaluate_only, a.model)