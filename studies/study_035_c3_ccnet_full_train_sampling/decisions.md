# Study 035 decisions

The primary comparison is the declared-budget final checkpoint after 2,048
updates. Selection overlaps 31,944 fit-covered traces and is retained only as a
training diagnostic. Its score cannot select a different primary checkpoint.

An earlier execution remains at
`runs/study_034_c3_ccnet_full_train_sampling/20260910T003711388064Z_6ad390c8b005_ccnet-train`.
It inherited the container's `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` because the
fresh-process runner did not apply the study's declared environment mapping.
That run and its 8.1139 dB prediction do not satisfy the Study 033 numerical
comparison condition and are excluded. The runner was corrected and checked
before the Study 035 preflight and primary execution. The accepted run records
`float32_matmul_precision=highest`, CUDA matmul TF32 disabled, cuDNN TF32
enabled, and cuDNN benchmark disabled, matching Study 033.

Candidate B performed worse than the control. The recorded diagnosis is limited
to the declared descriptors, training curve, coverage, and fixed-scale
distribution. The result did not trigger a seed retry, a full-training RMS fit,
epoch-wise patch resampling, or any other tuning.
