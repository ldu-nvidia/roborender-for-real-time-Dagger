# Policy checkpoints

Checkpoint names describe function rather than Raven's experiment-string or a
generic training step.

```text
checkpoints/
├── teachers/
│   ├── robot_policy/   # privileged state-to-action teacher for BC/DAgger labels
│   └── locomotion/     # fixed low-level walking and standing controllers
├── students/
│   ├── bootstrap/      # archived custom initialization; not Raven BC input
│   ├── BC/             # Raven offline-BC checkpoints and saved config
│   └── dagger_1..4/    # sequential full-state warm-start outputs
└── archive/
    ├── legacy_robot_teacher/
    └── legacy_student/
```

The active teacher is
`teachers/robot_policy/g1_privileged_action_teacher_wsdpt_step044500.pt`.
It maps privileged simulator observations to the 26-dimensional expert action
used as the ground-truth label. It is not a renderer and Isaac does not produce
that action by itself.

The retained custom student bootstrap is
`students/bootstrap/g1_3view_student_bootstrap_v1.pt`; its adjacent policy spec
binds vertically stacked head/left-hand/right-hand RGB, 113 actor-state values,
an 11-dimensional student output, and the action slice `[0,11]`. Raven-exact
offline BC does **not** load it: the policy is constructed fresh with the
configured ImageNet-pretrained ResNet18 vision module. DAgger round 1 loads the
resulting three-view BC checkpoint; later rounds load the immediately preceding
round's complete checkpoint.

The distributed DAgger learner publishes an immutable candidate after every
81-step × 28-environment update. Intermediate candidates advance the live
student between validation boundaries. At the configured 40-update boundary,
promotion is decided by fixed-panel closed-loop task success, not validation
MSE. W&B and the runtime checkpoint registry record the parent digest, training
MSE, gradient norm, candidate digest, and validation result.

Large candidate files and per-run registries remain local and ignored. The
tracked `manifest.json` identifies stable teacher/bootstrap inputs; accepted BC
and DAgger outputs carry adjacent runtime manifests and content hashes. See
`PROVENANCE.md` for lifecycle rules.
