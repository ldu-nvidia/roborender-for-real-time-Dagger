# Online Image3F DAgger

This stage is the H200/path-portable form of Raven commit `94d7985`'s
three-view Image3F extension. The only experiment-scale input is `num_envs`.

Every policy step captures synchronized current depth and robot-only RGB for
`head_color`, `hand_left`, and `hand_right`. Image3F renders the three views
before the policy acts. One teacher-generated anchor triplet is sampled for
each environment episode and remains FT-FIXED until that episode resets.

The measured Raven settings are fixed:

- teacher drives every action (`rollout_action_source: teacher`, beta 1.0);
- the teacher labels every visited state;
- 50% of environment episodes retain ray-traced observations and 50% use
  Image3F observations;
- seed 42, anchor index 0, h5_v4 depth, 240x416 renderer controls;
- Image3F five steps, CFG 1, TeaCache 0.3, untiled VAE;
- 81 live steps followed by one update with one minibatch;
- only the current live rollout is trained—there is no replay buffer;
- four rounds, each restoring the preceding checkpoint's model, optimizer,
  scheduler, environment, and trainer state, then running a full additional
  round budget.

Raven's exact trainer, PPO base, client, and Hydra config live in the clean
`../GR00T-VisualSim2Real` checkout at `94d7985`. Their identities and clean-tree
status are checked by `raven_image3f_dagger_contract.v1.json` and its validator;
no compatibility trainer or refactored resume path is used.

Start one or more renderer services with `run_image3f_server_h200.sh`, collect
their endpoint files in the colon-separated `IMAGE3F_ENDPOINT_FILES`
environment variable, then invoke `run_raven_dagger_round_h200.sh`. Round 1
must receive the selected three-view BC checkpoint; rounds 2-4 receive the
selected checkpoint from the immediately preceding round.

`run_raven_four_rounds_h200.sh` performs that exact four-round sequence and
hands each round's `last.pt` to the next. It exposes only the selected BC
checkpoint and `num_envs`; all scientific parameters remain fixed.

The online slot mapping is specifically `ext1=head_color`,
`ext2=hand_left`, `wrist=hand_right`. It must not be replaced by the offline
teacher's left/right/head request order.
