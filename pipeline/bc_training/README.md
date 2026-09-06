# BC training

Owns Raven-equivalent supervised fitting from
the flat `../../data/BC/rendered/chunk_*.npz` dataset.

- `train_raven_bc_h200.sh` launches Raven's literal one-process topology in the
  qualified H200 runtime. Its only scientific override is total `NUM_ENVS`.
- `wsdpt_videogen_offline_bc_3view.yaml` and
  `distill_trainer_videogen.py` come directly from the clean `94d7985` checkout;
  the launcher relocates only checkpoint, dataset, output, and runtime paths.
- Raven's offline branch performs cyclic sorted chunk loading,
  vertical `head_color,hand_left,hand_right` tiling, 81 steps, one minibatch,
  one optimizer update per loaded environment batch.

Scientific parameters remain Raven's: fresh ImageNet-pretrained trainable
ResNet18; vision feature dimension 128; SiLU MLP `[512,256,128]`; actor state
plus vision input; uniform L2/MSE on the first 11 dimensions of the 26-D
teacher action; coefficient 1; actor LR `2e-4`; constant scheduler; one
learning epoch; value, entropy and object-prediction loss coefficients zero;
frozen locomotion controllers and action std; 50,000 maximum batches; model
save every 200 batches.

The learner never reads validation or test paths. Raven did not use
validation-MSE saturation inside this training loop, so validation-based early
stopping is deliberately not part of this faithful stage.
