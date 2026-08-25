# Calibration data

The internal repository's `calibration/` directory holds ~59 MB of hand-eye
calibration captures for one physical camera rig. It is not part of this
release: the data describes specific camera bodies at specific mount points and
is meaningless for another robot. Nothing in the code requires it.

Calibration is per-rig and meant to be regenerated — the record / solve / apply
workflow is [`docs/SETUP.md`, section 5 "Hand-eye calibration"](docs/SETUP.md#5-hand-eye-calibration);
the tools are `humanoid_handeye_*.py` (listed in `docs/ARCHITECTURE_MAP.md`,
section 7). The data they write lands in `control/calibration/`, which the
recording step creates on first use; it is per-rig data and is not shipped.

`reference/` is likewise absent: it held a vendor motor manual PDF, which is the
manufacturer's document to distribute, not ours.
