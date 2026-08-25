"""auto_operator — typed components extracted from humanoid_auto_operator.py.

Stage-by-stage extraction per control/docs/auto_operator_refactor_plan.md.
The single-file operator remains the entry point and the behavioral source of
truth; every stage is gated by the behavioral suite and the differential
replay harness (see docs/auto_operator_characterization_plan.md).

``independent`` is the opt-in-at-constructor/default-at-CLI symmetric arm
scheduler.  It deliberately reuses the existing safety, staging, servo, hold,
grasp, and arbitration components instead of duplicating those contracts.
"""
