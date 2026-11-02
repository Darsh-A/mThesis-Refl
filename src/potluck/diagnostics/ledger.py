"""Rejection ledger summaries."""

from __future__ import annotations

from typing import Any


def ledger_fractions(meta: dict[str, Any]) -> dict[str, float]:
    """Return the fraction failing each cut and the acceptance rate."""
    ledger = meta["ledger"]
    drawn = float(ledger.get("drawn", 0))
    if drawn <= 0.0:
        return {
            key: 0.0
            for key in ("r1", "r2", "r3", "r4", "accepted", "accepted_total")
        }
    accepted = float(ledger.get("accepted", 0))
    return {
        "r1": ledger.get("r1", 0) / drawn,
        "r2": ledger.get("r2", 0) / drawn,
        "r3": ledger.get("r3", 0) / drawn,
        "r4": ledger.get("r4", 0) / drawn,
        "accepted": accepted / drawn,
        "accepted_total": (accepted + ledger.get("discarded", 0)) / drawn,
    }


def format_ledger(meta: dict[str, Any]) -> str:
    """Format the rejection ledger as a short table."""
    ledger = meta["ledger"]
    fractions = ledger_fractions(meta)
    lines = [
        f"drawn       {ledger.get('drawn', 0):>10d}",
        f"R1 fail     {ledger.get('r1', 0):>10d}  {fractions['r1']:.3f}",
        f"R2 fail     {ledger.get('r2', 0):>10d}  {fractions['r2']:.3f}",
        f"R4 fail     {ledger.get('r4', 0):>10d}  {fractions['r4']:.3f}",
        f"R3 fail     {ledger.get('r3', 0):>10d}  {fractions['r3']:.3f}",
        f"accepted    {ledger.get('accepted', 0):>10d}  {fractions['accepted']:.3f}",
    ]
    if ledger.get("discarded"):
        lines.append(
            f"accepted*   {ledger['accepted'] + ledger['discarded']:>10d}  "
            f"{fractions['accepted_total']:.3f}  (before target truncation)"
        )
        lines.append(f"discarded   {ledger['discarded']:>10d}  (accepted beyond target)")
    return "\n".join(lines)
