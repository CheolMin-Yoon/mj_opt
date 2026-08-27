# Agent Entry

이 저장소는 G1을 첫 대상으로 하는 **offline full-body trajectory compiler**를 소유합니다. 현재 구현
범위는 scheduled-contact inverse-dynamics direct transcription인 Level A이며, online MPC나 RL learner가
아닙니다.

## Boundary

- `source/`: 구현 source root. 내부 파일·폴더 구조는 사용자가 정하며 명시적 요청 없이 생성·재구성하지 않음
- `docs/`: architecture, formulation, artifact, development와 provenance 계약
- `docs/config/`: 실제로 소비되는 repository-local tool/problem configuration

공유 dynamics, contact와 Pinocchio manifold 식을 이 저장소에서 두 번째로 작성하지 않습니다. 승인된 CasADi
Function을 명시적인 source/model revision과 ABI로 소비합니다. MuJoCo state/action/clock은 adapter만 소유하고,
formulation과 solver가 simulator를 직접 step하지 않습니다.

내부 source taxonomy를 이 문서가 미리 정하지 않습니다. Online MPC, WBC/QP, HZD/contact-implicit와 RL learner는
해당 lifecycle을 사용자가 채택하기 전까지 범위 밖이며, `/home/frlab/mj_rl` 구현도 여기로 복제하지 않습니다.

## Required Route

1. 공통 project policy: `/home/frlab/research-wiki/AI-Sessions/wiki/harness/policies/project-policy.md`
2. 기본 응답 계약: `/home/frlab/research-wiki/.agents/skills/focus-output/SKILL.md`
3. 로컬 계약: `README.md`, `docs/README.md`
4. source portal: `/home/frlab/research-wiki/AI-Sessions/wiki/research/sources/mj-opt-active.md`
5. canonical method: `/home/frlab/research-wiki/AI-Sessions/wiki/research/methods/casadi-full-body-trajectory-optimization.md`
6. procedure: `/home/frlab/research-wiki/.agents/skills/casadi-trajopt/SKILL.md`
7. durable capture가 필요할 때만: `/home/frlab/research-wiki/prompts/reflect.md`

Native Pinocchio 의미·model prerequisite가 바뀌면 `pinocchio-core`, cpin named Function graph가 바뀌면
`casadi-pinocchio`를 먼저 적용합니다. Online horizon은 `casadi-mpc`, single-tick QP/HQP는 `casadi-opt`,
policy 소비는 target task repository로 라우팅합니다.

## Verification

```bash
python3 -m compileall -q source
python3 -m ruff check --config docs/config/pyproject.toml .
```

Solver 성공만으로 accepted trajectory라 부르지 않습니다. Raw status 보존, constraint-family별 canonical
residual, dense interval feasibility와 artifact round-trip을 모두 통과한 뒤 MuJoCo playback으로 넘깁니다.
