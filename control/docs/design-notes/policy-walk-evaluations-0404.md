# Walking policy evaluations, 2026-04-04

Hardware comparison of the PhaseEEDR checkpoint family against the no-phase
baseline, recorded the same afternoon. Migrated verbatim from `control/readme.md`
during the public-release cleanup; recording paths are relative to
`humanoid_site.RECORDINGS_DIR` (the takes themselves are not published).

Headline: **no checkpoint in this family recovers from pushes**, and the
no-phase baseline `HumanoidVelocityRMACNNShortEstimator` walked at least as well
as the best of them (v7) while stepping quieter.

recordings is in <control>/

the real HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv6 walks ok but the inital steping is unstable, the robot tend to unstably lean/tilt towards to the stance leg side, and it cannot recover from pushes, it is not reactive
real data saved to <recordings>/HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv6_20260404_175237.pkl


the real HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv7 walks ok, the inital steping is better than v6, as it does not overly lean towards the stance leg side. it seems to step quieter than v6. still v7 cannot recover from pushes. the real data is saved to <recordings>/HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv7_20260404_183449.pkl

the real HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv8 seems to be similar but weaker than v7, the foot seems to drag on the ground more than v7. v8 cannot recover from pushes. the real data is saved to  <recordings>/HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv8_20260404_184029.pkl

the real HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv9 feels similar to v7. v9 cannot recover from pushes. v9 real data is <recordings>/HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv9_20260404_184352.pkl

the real HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv10 feels similar to v8, feels a bit week, cannot recover from pushes, v10 real data is <recordings>/HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv10_20260404_184902.pkl

the real HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv11 seems weak, cannot recover from pushes, side walking (y velocity tracking) is unstable. the v11 real data is <recordings>/HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv11_20260404_185158.pkl

the real HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv12 is the worst, it does not follow command velocity at all, it only stands, and only rotate with one leg. the v12 real data is <recordings>/HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv12_20260404_185459.pkl

as a comparison the baseline no phase signal no arm override real HumanoidVelocityRMACNNShortEstimator walks ok, steps quieter, and recovers from pushs slightly better than HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv7. the HumanoidVelocityRMACNNShortEstimator real data is <recordings>/HumanoidVelocityRMACNNShortEstimator_20260404_185928.pkl


