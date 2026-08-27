# Formulation contract

## First problem class

첫 문제는 fixed scheduled-contact, full-body inverse-dynamics direct transcription인 Level A입니다. Online MPC,
contact-implicit optimization, impact/reset과 HZD periodic gait가 아닙니다.

초기 paper comparison은 constant interval duration을 사용합니다. Time grid는 artifact에 보존하되 첫 구현에서
`dt`를 decision variable로 만들지 않습니다.

## Node and interval topology

`N`개 interval에는 `N+1`개 state node가 있습니다.

```text
state nodes k=0...N:       q_repr[k], v[k]
intervals   k=0...N-1:     a[k], active contact force/position, dt[k]
terminal node k=N:         q_repr[N], v[N] only
```

`q_repr`의 차원과 valid domain은 parameterization이 소유합니다. 모든 node는 유효한 physical Pinocchio
`q(nq)`로 decode되어야 하며 `v/a`는 `nv` tangent quantity입니다. Terminal acceleration, force, contact position,
unused time decision과 next state가 없는 transition row를 만들지 않습니다.

## Shared physical constraints

Matched representations는 같은 versioned Function bundle을 사용합니다.

- configuration transition과 physical decode
- whole-body terms와 inverse-dynamics residual
- stance forward kinematics와 no-slip/contact-position continuity
- unilateral, friction과 force-limit margins
- reconstructed torque 및 joint/velocity/torque bounds

Project-facing contact force는 world-frame point force로 고정합니다. Contact-local donor oracle을 비교할 때는
world rotation과 `Jᵀf` sign을 non-identity foot orientation에서 검증해야 합니다. Joint/frame/contact order와
active mask는 pack, solve, audit와 artifact에서 동일해야 합니다.

## Solver seam

Formulation은 decision/parameter slice, objective/row order, bounds, scaling, initialization과 decode map을
소유합니다. Numerical adapter는 plugin workspace, options, primal/dual state와 raw status를 소유합니다.
Exception, iteration limit와 infeasible iterate를 성공으로 바꾸지 않습니다.

Artifact 및 post-solve acceptance는 [artifact contract](artifact.md)가 소유합니다.
