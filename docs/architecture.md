# Architecture

## Decision

`mj_opt`는 **offline compiler, trajectory acceptance와 MuJoCo handoff contract**를 소유합니다. `source/` 내부
taxonomy는 이 문서가 정하지 않습니다.

```text
native Pinocchio oracle
        ↓ parity
accepted named CasADi Functions
        ↓
floating-base parameterization → shared offline formulation → solver adapter
                                                        ↓ raw result/status
                                             canonical validation
                                                        ↓
                                             trajectory artifact
                                                        ↓
                                               MuJoCo adapter
                                                        ↓
                                        later: mj_rl tracking consumer
```

## Logical ownership

| logical layer | owns | does not own |
| --- | --- | --- |
| parameterization | solver coordinate, physical configuration decode, difference/integration와 valid domain | dynamics, contact, solver status |
| physical Function seam | versioned CasADi ABI와 native parity evidence | horizon packing, simulator step |
| offline formulation | `N/N+1` layout, costs/constraints, bounds, scaling와 initialization semantics | plugin workspace, MuJoCo clock |
| solver adapter | NLP mapping, options, numerical warm state와 raw diagnostics | physical acceptance |
| validation/artifact | canonical residual, dense audit, accepted/best-infeasible와 physical serialization | reward, learner |
| MuJoCo handoff | model/order/time conversion, reset/playback와 requested/applied torque trace | equation or solver reassembly |

## Matched comparison

첫 reproduction은 SE(3) tangent, quaternion 세 변형과 RPY를 비교합니다. 비교마다 바뀌는 것은 floating-base
decision, difference와 elementary integration뿐입니다. Robot model, physical equations, task cost, contact schedule,
initialization, solver options와 acceptance threshold는 고정해야 합니다.

각 표현은 같은 physical Pinocchio `q(nq)`와 tangent `v(nv)`를 shared Function seam에 제공해야 합니다. 표현별
RNEA/contact 식을 따로 작성하거나 한 표현에만 유리한 warm start·termination rule을 두면 matched comparison이
아닙니다.

## Dependency rules

Solver는 MuJoCo를 step하지 않고, MuJoCo는 solver 내부 배열을 해석하지 않습니다. MuJoCo는 accepted artifact만
소비합니다. `/home/frlab/mj_rl`은 이후 task/learner consumer이며 `mj_opt`는 `mj_rl`이나 RSL-RL을 import하지
않습니다.

첫 범위는 scheduled-contact Level A입니다. Impact/reset/periodicity/HZD, contact-implicit discovery, online MPC,
single-tick WBC/QP와 generated evaluator는 사용자가 별도 lifecycle로 채택하기 전까지 비범위입니다.
