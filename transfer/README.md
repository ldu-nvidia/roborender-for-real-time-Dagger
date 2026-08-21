# Applying the private pipeline to a fresh GR00T checkout

All four patches apply on top of public NVlabs revision 92bf0863. The first three are the
original feature commits; 0004 is a cumulative diff from the exact tree produced by 0001--0003
to the pinned production integration:

    cd GR00T-VisualSim2Real
    git am ../transfer/0001-*.patch ../transfer/0002-*.patch ../transfer/0003-*.patch
    git apply ../transfer/0004-*.patch

0001 adds the videogen pipeline: the trainer (collect / offline / online / image_dagger
modes), the Image3F and Raven render clients, conditioning cameras and scene walls in the
simulator, and the exp configs.

0002 adds three-view policy support: cameras.policy_views + policy_view_count tile the
views vertically into one vision_obs, with camera_resolution and the history buffer scaled
to match.

0003 adds the explicit DAgger rollout action source and teacher/student beta schedule.

0004 adds portable runtime paths, three-view enforcement, restart-complete checkpoints,
DAgger telemetry, Image3F client hardening, Isaac camera/lifecycle gates, fixed-seed evaluation
evidence, and exact policy-parameter checkpoint digests. Commit the resulting GR00T tree locally
before building an HSG snapshot so its 40-hex revision can be recorded in `.source_revision`.

Not included, because they live outside this repo:

  - launchers and docs -> fast_roborender/ (scripts/, README.md)
  - Image3F bundle + adapter, foundation weights, TensorRT engines (arch-specific, built
    locally; see the Setup section of fast_roborender/README.md)
  - the anchor pool and the rendered dataset

Run fast_roborender/scripts/setup/check_env.sh on the target machine first; it reports
which of those are missing.
