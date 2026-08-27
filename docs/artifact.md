# Trajectory artifact contract

## Required physical fields

Accepted artifact는 최소한 다음을 보존합니다.

- physical `q(nq)`, `v`, interval `a`, `f_world`, `p_world`, reconstructed `tau`와 `dt`
- solver-coordinate trajectory와 parameterization identifier
- joint, frame, contact, phase와 time order 및 active-contact mask
- frame, unit, interpolation, model/profile/source revision과 config hash
- CasADi, Pinocchio, solver와 adapter version
- objective, unmodified raw solver status/statistics와 family별 canonical residual
- `accepted_feasible` 또는 `best_infeasible`, dense violation과 refinement history

Solver-private slack, dual 또는 collocation auxiliary가 재현에 필요하면 별도 internal section에 저장하며 physical
control처럼 노출하지 않습니다.

## Acceptance

Serialization 전에 canonical Function으로 variable domain, dynamics, contact, torque, boundary와 task residual을
다시 계산합니다. Node 사이 trajectory를 dense sampling하고 write/read 및 joint/frame/contact/time round-trip을
검사합니다. Solver `success=True`만으로 accepted artifact를 만들지 않습니다.

## MuJoCo handoff

MuJoCo adapter는 artifact의 model hash, joint/actuator order, timestep/hold cadence, initial state와 friction/limit
profile을 target MJCF와 대조합니다. First gate는 kinematic/state-reference playback이고, torque-applied open-loop와
controller tracking은 그 다음의 서로 다른 gate입니다.

MuJoCo에서의 접촉 차이 또는 tracking 실패는 optimizer artifact를 자동 무효화하지 않으며, 반대로 playback이
보인다는 사실도 canonical NLP feasibility를 대신하지 않습니다.
