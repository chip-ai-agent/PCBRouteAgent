"""Budget accounting: executor facade calls, local retries, Global replans.

Counts are *facade* calls (one ``NativeAdapter`` method = 1), not C++ primitives — an
aggregate getter may cost several native calls underneath. LLM tokens and wall time are
recorded separately by the providers. Nothing is ever refunded by a rollback.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Budget:
    executor_calls: int = 400
    replan_cap: int = 3
    local_retry_cap: int = 3
    reserve_calls: int = 8                      # kept back for recovery (cancel / rollback / save)
    spent_calls: int = 0
    replans: int = 0
    retries: dict = field(default_factory=dict)  # edge_id -> local retries used
    history: list[dict] = field(default_factory=list)

    @classmethod
    def from_plan(cls, plan_budget) -> "Budget":
        return cls(executor_calls=int(plan_budget.executor_calls), replan_cap=int(plan_budget.replan_cap),
                   local_retry_cap=int(plan_budget.local_retry_cap))

    @property
    def remaining_calls(self) -> int:
        return self.executor_calls - self.spent_calls

    def charge(self, n: int = 1, what: str = "") -> None:
        if self.spent_calls + n > self.executor_calls - self.reserve_calls:
            raise BudgetExhausted(f"executor budget exhausted ({self.spent_calls}+{n} > "
                                  f"{self.executor_calls - self.reserve_calls}) at {what}")
        self.spent_calls += n

    def charge_recovery(self, n: int = 1) -> None:
        """Recovery calls may dip into the reserve but never past the hard cap."""
        if self.spent_calls + n > self.executor_calls:
            raise BudgetExhausted("hard stop: even the recovery reserve is spent")
        self.spent_calls += n

    def can_retry(self, edge_id: str) -> bool:
        return self.retries.get(edge_id, 0) < self.local_retry_cap

    def use_retry(self, edge_id: str) -> int:
        self.retries[edge_id] = self.retries.get(edge_id, 0) + 1
        return self.retries[edge_id]

    def can_replan(self) -> bool:
        return self.replans < self.replan_cap

    def use_replan(self, reason: str = "") -> int:
        if not self.can_replan():
            raise BudgetExhausted(f"replan cap {self.replan_cap} reached")
        self.replans += 1
        self.history.append({"replan": self.replans, "reason": reason, "spent_calls": self.spent_calls})
        return self.replans

    def revise(self, **changes) -> None:
        """Explicit budget revision from a GlobalGuidePatch; spent counters are untouched."""
        for k, v in changes.items():
            if k in ("executor_calls", "replan_cap", "local_retry_cap", "reserve_calls"):
                setattr(self, k, int(v))

    def snapshot(self) -> dict:
        return {"executor_calls": self.executor_calls, "spent_calls": self.spent_calls,
                "remaining_calls": self.remaining_calls, "replans": self.replans,
                "replan_cap": self.replan_cap, "retries": dict(self.retries),
                "local_retry_cap": self.local_retry_cap}
