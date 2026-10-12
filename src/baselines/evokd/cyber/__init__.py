"""Paper-faithful EvoKD reproduction for Medicine QA (Algorithm 1, EvoKD.pdf).

Deliberately independent of studies/evokd_chemprot_reproduction_v1/evokd/:
no admission/rejection filter of any kind is imported or reimplemented
here, and generation is one batched call per chat event, never per-root.
See DESIGN_v1.md for the full paper-to-code mapping.
"""
