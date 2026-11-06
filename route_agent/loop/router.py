"""Feedback policy router: what happens after a Detailed step reports.

Mirrors the design table:

    feedback                              next
    progress                              continue the task
    local target / mode failure           retry inside the guide (anchor / mode / waypoint), capped
    same corridor repeatedly no progress  end branch -> Global replan on the affected edge
    transition / pin access unavailable   -> Global replan (alternate_layer / widen)
    several nets contend for a channel    -> Global replan (reorder / reservations / repair scope)
    budget or replan cap reached          hard stop

The router is pure: it reads a FeedbackPacket and the Budget and returns a Decision; the
orchestrator performs rollback / provider calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from route_agent.loop.budget import Budget
from route_agent.schema.feedback import FeedbackPacket


@dataclass
class Decision:
    kind: str                     # continue | retry | replan | stop | done
    reason: str
    request: str = ""             # for replan: widen | alternate_layer | reorder | repair
    edge_id: str = ""
    hints: dict = field(default_factory=dict)


@dataclass
class StallTracker:
    """Counts consecutive no-progress failures per (edge, stage/corridor)."""
    threshold: int = 2
    counts: dict = field(default_factory=dict)

    def record(self, edge_id: str, region: str, progressed: bool) -> int:
        key = (edge_id, region)
        if progressed:
            self.counts[key] = 0
            return 0
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def stalled(self, edge_id: str, region: str) -> bool:
        return self.counts.get((edge_id, region), 0) >= self.threshold


def stall_region(pkt: FeedbackPacket) -> str:
    """The one key progress and failure counters share: the region the Detailed role was in
    (failure evidence first, then the stage it reported)."""
    return pkt.failure_evidence.region_ref or pkt.stage_ref


def route_feedback(pkt: FeedbackPacket, budget: Budget, stalls: StallTracker) -> Decision:
    eid = pkt.edge_id
    region = stall_region(pkt)
    if pkt.outcome == "done":
        return Decision("done", "edge connected", edge_id=eid)
    if pkt.outcome == "progress":
        stalls.record(eid, region, True)
        return Decision("continue", "connection progressed", edge_id=eid)

    if pkt.outcome == "unsupported":
        if not budget.can_replan():
            return Decision("stop", "unsupported edge and replan cap reached", edge_id=eid)
        return Decision("replan", "executor lacks a capability the guide needs", request="alternate_layer", edge_id=eid)

    if pkt.outcome == "conflict":
        if not budget.can_replan():
            return Decision("stop", "resource conflict and replan cap reached", edge_id=eid)
        return Decision("replan", "nets contend for the same channel", request=pkt.request or "reorder",
                        edge_id=eid, hints={"nets": [b.get("net_code") for b in pkt.failure_evidence.blocker_candidates]})

    # blocked / partial
    n = stalls.record(eid, region, False)
    reason = pkt.failure_evidence.native_reason
    # Only a via the engine refused to place is a transition failure that no local retry can fix.
    # A boxed-in head that merely ASKS for another layer (request alternate_layer with via_sites /
    # blockers attached) still gets the free local retry / rip-up first; the evidence travels with
    # the replan when that fails too. Replanning such edges at once burned the per-board replan cap
    # before the hardest edges were reached (D3-B Claude run, 2026-10-30: +30% Global calls, no
    # more edges).
    transition_failure = reason in ("via_on_thru_pad", "via_not_placed")
    if transition_failure or stalls.stalled(eid, region) or not budget.can_retry(eid):
        if not budget.can_replan():
            return Decision("stop", f"no progress in {region} and replan cap reached", edge_id=eid)
        req = pkt.request or ("alternate_layer" if transition_failure else "widen")
        return Decision("replan", f"{n} consecutive failure(s) in {region}; native: {reason}", request=req,
                        edge_id=eid, hints={"region": region, "last_head": pkt.last_head})
    return Decision("retry", f"local failure in {region} (retry {budget.retries.get(eid, 0) + 1}/{budget.local_retry_cap})",
                    edge_id=eid, hints={"region": region, "native_reason": reason})
