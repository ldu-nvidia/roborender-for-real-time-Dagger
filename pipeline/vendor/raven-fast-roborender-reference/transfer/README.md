# Applying the private pipeline to a fresh GR00T checkout

All three patches apply on top of public NVlabs revision 92bf0863 (verified an ancestor of
the branch they were cut from), so no rebase is needed:

    cd GR00T-VisualSim2Real
    git am ../transfer/0001-*.patch ../transfer/0002-*.patch ../transfer/0003-*.patch

0001 adds the videogen pipeline: the trainer (collect / offline / online / image_dagger
modes), the Image3F and Raven render clients, conditioning cameras and scene walls in the
simulator, and the exp configs.

0002 adds three-view policy support: cameras.policy_views + policy_view_count tile the
views vertically into one vision_obs, with camera_resolution and the history buffer scaled
to match.

0003 adds the four-round DAgger beta schedule used by the distributed pipeline.

Not included, because they live outside this repo:

  - launchers and docs -> fast_roborender/ (scripts/, README.md)
  - Image3F bundle + adapter, foundation weights, TensorRT engines (arch-specific, built
    locally; see the Setup section of fast_roborender/README.md)
  - the anchor pool and the rendered dataset

Run fast_roborender/scripts/setup/check_env.sh on the target machine first; it reports
which of those are missing.
