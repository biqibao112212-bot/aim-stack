# H2 Analysis

## Negative smoke and correction

The first immediate pose-loss smoke (`dense-smoke-v19-laplace-epro-gpu`) produced an invalid large negative normalizer because random UV correspondences had not yielded a trustworthy online MAP mode. Its checkpoint is negative evidence and is not eligible for selection.

The loss now requires the online MAP energy to be no worse than labelled-pose energy before a sample contributes EPro normalization, and clamps the local Hessian log-determinant contribution. The repeated `dense-smoke-v19-consistent-epro-gpu` run stayed finite:

- Samples: 16 train / 16 exploratory validation
- Checkpoint SHA-256: `de6300ed578fa6e8d7e7a7c9f8bb7ed200df2245177db12646d86bd849faba07`
- Validation invalid MAP fraction: 68.75%

This supports the pre-registered curriculum: learn nominal projected support and UV first, then enable pose probability. It does not yet support an accuracy claim.
