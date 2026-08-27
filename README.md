# mj_opt

`mj_opt`는 G1 whole-body motion을 생성하는 offline trajectory-optimization compiler입니다. 첫 목표는
*A Comparative Study of Floating-Base Space Parameterizations for Agile Whole-Body Motion Planning*의 공정한
parameterization 비교를 CasADi/Pinocchio로 clean-room 재구성하고, accepted trajectory를 MuJoCo에서 검증하는
것입니다.

현재 implementation gate와 acceptance boundary는
[`docs/README.md#current-gate`](docs/README.md#current-gate)가 단독으로 소유합니다.

```text
accepted Pinocchio/CasADi Functions
                 │
                 ▼
parameterization → shared Level-A formulation → solver adapter
                                                │
                                                ▼
                            residual audit → trajectory artifact
                                                │
                                                ▼
                                      MuJoCo playback adapter
                                                │
                                                ▼
                                    later: mj_rl tracking consumer
```

## Layout

```text
source/                 implementation root; 내부 구조는 사용자가 결정
docs/                   설계·계약·개발 순서·근거의 진입점
docs/config/            repository-local environment/tool/problem configuration
```

Source 내부 taxonomy와 파일명은 이 scaffold가 정하지 않습니다. 논리적 책임 경계와 구현 순서는
[`docs/README.md`](docs/README.md)에서만 안내합니다.

## Development check

정확한 현재 preflight와 base solver environment는
[`docs/development.md#current-preflight`](docs/development.md#current-preflight)와
[`docs/config/environment.yml`](docs/config/environment.yml)이 소유합니다. Root README에는 실행 명령을 복제하지
않습니다. 이 environment의 CasADi/cpin/IPOPT gate 통과는 G1 trajectory solve나 model equivalence를 뜻하지
않습니다. Reference code와 robot meshes를 이 repository로 복사하지 않습니다.
