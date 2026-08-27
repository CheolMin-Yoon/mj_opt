# Documentation

이 폴더는 `mj_opt`의 구현 파일 배치가 아니라 **무엇을 구현하고 어떻게 판정할지**를 소유합니다.
`source/` 내부 구조와 실제 module 이름은 사용자가 정하며, 문서가 선행해서 강제하지 않습니다.

## Reading order

| 순서 | 문서 | 질문 |
| --- | --- | --- |
| 1 | [architecture](architecture.md) | optimizer, solver, artifact와 MuJoCo의 책임은 어떻게 나뉘는가? |
| 2 | [G1 model parity](g1-model-parity.md) | donor URDF와 candidate MJCF는 같은 optimizer model인가? |
| 3 | [formulation](formulation.md) | 첫 Level-A NLP의 state, interval, contact와 비교 조건은 무엇인가? |
| 4 | [artifact](artifact.md) | 어떤 결과만 accepted trajectory로 저장하고 MuJoCo에 넘기는가? |
| 5 | [development](development.md) | 어떤 순서로 구현·검증하며 어디서 멈춰야 하는가? |
| 6 | [provenance](provenance.md) | 어떤 논문·reference·wiki source를 어떻게 재사용하는가? |

## Current gate

현재 active gate는 pinned donor G1 URDF와 approved `/home/frlab/anaconda3/envs/mj_opt`를 사용하는 SE3 Level-A
one-interval versioned named-Function ABI acceptance입니다. `source/`는 아직 비어 있고, `tests/`의 native/cpin
회귀는 exploratory primitive probe이므로 project-consumed ABI acceptance가 아닙니다. Raw donor URDF-MJCF
drop-in parity는 기각됐고 normalized model은 diagnostic 후보로만 남습니다.

이 gate를 통과하기 전에는 `N/N+1` packing, deterministic G1 solve, trajectory artifact 또는 MuJoCo acceptance로
진행하지 않습니다. 정확한 순서와 실행 명령은 [development](development.md)가 소유합니다.

새 코드가 생기면 code-coupled shape, CLI와 test command는 이 repository가 소유하고, 여러 project에서 재사용할
수식·비교 결론만 research wiki 정본으로 승격합니다.
