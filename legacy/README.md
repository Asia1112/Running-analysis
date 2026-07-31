# Legacy compatibility layer

The modules under `src/running_biomechanics` are retained as a compatibility
layer for the previous broad running-biomechanics prototype. They include
cadence, hip, trunk, arm, symmetry, overstriding, vertical-motion, and general
technique-reporting features.

They are not imported by the current Streamlit application or the canonical
validation CLI. Keeping them in place preserves earlier experiments and their
tests without mixing them into the new `src/knee_angle_validation` package.

After archived pilot results have been reproduced, this package and its two
legacy tests can be moved physically under this directory in a separate,
reviewable change.
