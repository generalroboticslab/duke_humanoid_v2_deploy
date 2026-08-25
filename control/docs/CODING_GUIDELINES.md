# Coding guidelines

The conventions this codebase already follows. They are recorded so a change
reads like the code around it, not so anyone rewrites working code to match.

## Priorities

* Design first: before a change lands, its constraints (rates, units, frames,
  who owns which wire) are written down, in the docstring or the design note.
* Correctness and efficiency over cleverness. No overengineering.
* Guard against failures that have happened or can be shown to happen on this
  robot, not hypothetical ones. Speculative defensive code hides real faults and
  this stack has enough real ones — `docs/auto_operator_incidents.md` is the list.

## Structure

* Core logic is a single linear flow. Extract a function only for logic that is
  reusable or genuinely distinct, never to make a block look shorter.
* Clean structure, consistent naming; explicit names over short ones.
* Remove dead code, unused imports and unreachable branches when you can prove
  they are unreachable — grep plus a test run, not intuition.
* Named constants for magic strings and numbers used in more than one place.
* Single-responsibility functions. Catch specific exceptions; never bare
  `except`.
* Prefer a dataclass or NamedTuple over a dict once a shape has a name.

## Comments and docstrings

This is the part that matters most here. The comments in this codebase encode
hardware facts that cost robot time to learn.

* Docstrings state purpose, inputs and outputs, assumptions, and the design
  decision behind the code — what it does, why, and which alternatives were
  rejected. A reader arriving cold should not have to re-derive the reasoning.
* Inline comments explain the non-obvious: unit seams, sign conventions,
  ordering that is load-bearing, and the incident a guard exists to prevent.
* Keep the WHY next to the code. When a constant accumulates a long chronology
  of past values, move the ledger to `docs/design-notes/<topic>.md` and leave
  the current value's reason plus a pointer — see
  [design-notes/journey-walk-and-aim.md](design-notes/journey-walk-and-aim.md)
  for the pattern. Dates and run references are preserved, never summarised away.
* A comment that says a value is provisional, uncommitted or about to change
  must be removed once that stops being true. A stale banner is worse than none.

## Safety-critical changes

Any change to a control constant, a wire-protocol field, a CLI flag, a timing
budget or a safety gate changes what a 36 kg humanoid does with people beside
it. Such a change needs a hardware run behind it, and the measurement recorded
next to the value. `docs/auto_operator_incidents.md` lists the mechanisms that
exist because something went wrong; each entry names the "simplification" that
would reintroduce the failure.
