# Provenance and reuse boundary

## Primary semantic donor

- repository: `upatras-lar/se3_trajopt`
- local read-only clone: `/home/frlab/reference/se3_trajopt`
- checked revision: `1bbadc9573b2989a0f414888d4fa4af137d57db9`
- license: BSD-2-Clause
- paper: *A Comparative Study of Floating-Base Space Parameterizations for Agile Whole-Body Motion Planning*,
  Humanoids 2025, arXiv `2508.11520`

The donor is a semantics and numerical-comparison oracle. Its node packing, handwritten derivative callbacks,
partial JSON lifecycle, examples and tests are not transplanted. No G1/Go2/TALOS URDF, MJCF or mesh has been copied;
the bundled asset files do not carry a separate provenance/license record.

## Canonical local primitives

Reusable physical expressions currently live in `/home/frlab/research-wiki/casadi/`, with source provenance in
`casadi/manifest.yaml`. The initial relevant seams are:

- `trajectory_optimization/manifold_state.py`
- `dynamics/whole_body_terms.py`
- `dynamics/whole_body_dynamics.py`
- `dynamics/contact_constraints.py`

Presence in that source store is not acceptance. Before `mj_opt` consumes a Function bundle, its exact source/model
revision, ABI, `nq/nv`, frame/order/sign/units, valid domain, derivatives and independent native parity must be
recorded and tested in this project.

## Repository status

This project was initialized locally from an empty `/home/frlab/mj_opt` directory on 2026-08-26. It currently has no
remote, checked implementation commit or selected redistribution license. Until a project license is chosen, do not
redistribute this repository. If donor code is ever copied rather than independently re-authored, retain the exact
BSD notice and record file-level provenance before distribution.

The `source/` directory is intentionally empty at the scaffold gate. Logical responsibilities in the other docs do
not prescribe future module names or directories.
