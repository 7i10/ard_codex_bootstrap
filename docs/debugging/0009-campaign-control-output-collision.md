# Campaign control/output namespace collision

## Failure

The first fixed-SHA campaign attempt (`23968e7f7c43121676de5c0937187777184de285`) failed all three training pilots
before model construction. `CampaignWorker` placed detached phase records at
`<JOB_OUTPUT_DIR>/campaign-control/train`; creating that parent made the scientific output directory exist before
`ard.cli.train` acquired it. The train output guard correctly rejected overwrite without `--resume`.

The first observed exception was:

```text
FileExistsError: refusing to overwrite existing output directory without --resume
```

No training batch, attack, optimizer update, W&B run, or scientific metric was produced. The attempt is retained as
failed operational evidence and is not a pilot result.

## Root cause and fix

Campaign orchestration and scientific output had overlapping ownership. Phase launch/exit/adoption records now live
under `<campaign-state>/phases/<job>/<phase>`, while the train/evaluate CLI remains the sole creator of
`<JOB_OUTPUT_DIR>`. The output collision guard was not weakened.

Regression coverage asserts that a train launch can create its campaign control path without creating the scientific
output directory. Because this changes the campaign runtime, all three pilots must run again from the new pushed SHA;
no evidence from the failed SHA is reusable for the production gate.
