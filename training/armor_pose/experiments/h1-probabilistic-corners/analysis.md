# H1 Analysis

## Implementation smoke

- Run: `sparse-smoke-v19-laplace-epro-gpu`
- Samples: 16 train / 16 exploratory validation
- Checkpoint SHA-256: `12f31ed9edea8fd719cb258bb129bd0d0ae361b540d4bfb0c15e6effa51b6107`
- Outcome: grid, context, tail NLL, final corner NLL, Laplace-EPro and physical terms remained finite through CUDA backward.
- Limitation: one epoch and 16 samples cannot support an accuracy conclusion; invalid MAP fraction was 25%.

The next confirmatory step is a curriculum-free sparse pilot on the larger non-test development pack, followed by a frozen P95 evaluator. V15/V18 remain inaccessible.
