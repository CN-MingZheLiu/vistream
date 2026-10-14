# Evaluation source

The unchanged `verify_results.py` implements separate generation sealing,
ground-truth evaluation and numerical verification. The original metric source,
fixed target values, contract documentation and synthetic integrity tests are
included. No prediction arrays or ground truth are included.

Read `../docs/SETUP_EN.md` before attempting to execute the full workflow. This
source-only release lacks the original runtime and weights. `target_results.json`
contains reference scores and support metadata, not inference inputs.

The latest complete binary-bundle validation scored 15.25 m / 9.10 m mean KITTI
Sim(3) ATE for loop off/on. It did not exactly reproduce the earlier repaired
15.08 m / 10.41 m target, so its status remains `NOT_REPRODUCED`.

Format/integrity tests are synthetic and do not validate a GPU run:

```bash
python -m unittest discover -s evaluation -p test_validation_contract.py -v
```

Evaluation requires all original sequence support, genuine stage receipts and a
complete unchanged seal before it opens external GT. Do not bypass those checks
to score an incomplete run. See `GENERATION_CONTRACT.md` for the original schema.
