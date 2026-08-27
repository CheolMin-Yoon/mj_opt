# G1 URDF-MJCF model parity audit

> **판정 (2026-08-26):** donor URDF와 MuJoCo Menagerie MJCF는 joint topology는 같지만, 원본 그대로는
> trajectory-optimization 관점의 drop-in equivalent가 아니다. 물리 profile과 contact frame을 맞추면 수치가
> 근접하지만 bit-identical하지 않다. 또한 SE3_TrajOpt의 저장된 G1 trajectory artifact가 없어서 **원 궤적 결과의
> 동일성은 아직 판정할 수 없다.**

이 문서는 구현 source가 아니라 G1 model 선택과 acceptance gate를 소유한다. 실행 입력과 관측값의 machine-readable
snapshot은 [`config/g1-model-parity.toml`](config/g1-model-parity.toml)에 있다.

## 논문 근거

Wiki에는 official `SE3_TrajOpt@1bbadc9` source portal이 checked 상태로 있지만 paper note는 아직 ingest되지 않았다.
Checked source는 G1 `(q_pin, q_tan, v, a)=(36,35,35,35)`와 fixed-contact full inverse dynamics를 사용한다. 그러나
repository에는 논문의 G1 optimized trajectory가 없고, JSON saver도 reconstructed torque와 contact position을
완전하게 보존하지 않는다.

**우리의 도출:** parameterization이나 solver 결과를 비교하기 전에 URDF와 MJCF가 같은 physical primitive를
정의하는지를 독립 gate로 검사해야 한다. 이 audit은 그 model gate만 다루며 논문 trajectory 재현을 주장하지 않는다.

## 검사할 주장

> 동일한 physical configuration, tangent velocity/acceleration과 contact definition을 입력했을 때, 선택한 MJCF의
> native Pinocchio primitive는 donor URDF oracle과 사전에 정한 허용오차 안에서 일치한다.

현재 결과는 다음과 같다.

- **raw MJCF:** 기각. Armature, friction, velocity/base bounds와 contact-frame surface가 다르다.
- **in-memory normalized MJCF:** 근접하지만 diagnostic-only. 관측 후 tolerance를 정하지 않았으므로 accepted
  equivalent가 아니다.
- **원 trajectory 동일성:** 판정 보류. 원본 trajectory artifact가 없다.

## 정답 기준

Model oracle은 donor repository에 포함된 URDF의 native Pinocchio 4.1.0 결과다. 비교 tensor는 joint placement,
contact-frame placement/Jacobian, mass matrix `M(q)`, nonlinear effects `nle(q,v)`, inverse dynamics
`rnea(q,v,a)`와 COM이다.

두 가지 내부 parity를 먼저 분리한다.

1. 각 native model과 `cpin.Model(native_model)`이 동일한 configuration에서 일치해야 한다.
2. 같은 MJCF를 읽은 MuJoCo와 native Pinocchio가 quaternion을 명시적으로 재배열한 뒤 `M`과 zero-velocity bias에서
   일치해야 한다.

Full trajectory의 정답은 serialized `q/v/a/contact force/contact position`, contact schedule, terrain, time grid,
model revision과 solver status를 포함한 원 artifact를 donor URDF에서 다시 residual-audit한 값이다. 이 artifact가
없는 현재의 model probe는 trajectory equality의 대체물이 아니다.

## 핵심 객체·신호의 의미

- `raw error`: 두 loader가 원본 파일을 그대로 읽었을 때 같은 tensor의 최대 절대차다. 0이면 표본에서 같다는
  뜻이며, 작다는 사실만으로 전체 domain이나 최적해 동일성을 보장하지 않는다.
- `normalized error`: MJCF native model의 dynamics/limit field를 donor URDF 값으로 임시 복사하고 missing contact
  frame을 합성한 뒤의 최대 절대차다. Model conversion 후보의 오차이지 MuJoCo XML 자체가 같다는 뜻이 아니다.
- `native-cpin error`: parser 결과를 CasADi graph로 cast하는 과정의 오차다. URDF-MJCF 간 model 차이를 측정하지
  않는다.
- `trajectory parity`: 같은 NLP semantics와 model profile에서 accepted artifacts의 canonical residual과 trajectory를
  비교하는 별도 gate다. Solver의 `success` 문자열이나 비슷한 animation만으로 통과하지 않는다.

## 적용 위치

이 audit은 optimizer 앞의 **model/reference preprocessing gate**에 적용한다. 권장 경로는 다음과 같다.

```text
URDF or MJCF --native Pinocchio parser--> accepted native Model
                                          |
                                          +--> geometry/contact-frame audit
                                          |
                                          +--> cpin.Model(native Model)
                                                 |
                                                 +--> trajectory formulation
```

`pinocchio.casadi`에는 URDF/MJCF parser와 GeometryModel이 없다. 따라서 MJCF도 native
`buildModelFromMJCF`로 읽은 뒤 `cpin.Model`로 cast한다. `buildModelsFromMJCF` shortcut을 쓸 때는 contact model
추가 여부를 암묵값에 맡기지 말고 `contacts=False`를 명시한다.

## 선택한 model과 provenance

| 역할 | 파일과 revision | SHA-256 | Pinocchio `(nq,nv,njoints,nframes)` | 질량 |
| --- | --- | --- | --- | --- |
| URDF oracle | `se3_trajopt@1bbadc9`, `g1_29dof.urdf` | `d44063d6...fa775` | `(36,35,31,112)` | `33.34114202 kg` |
| MJCF candidate | `mujoco_menagerie@71f066a`, `unitree_g1/g1.xml` | `3c261655...647f3` | `(36,35,31,65)` | `33.34114200 kg` |

두 model의 29개 actuated joint 이름, 순서와 parent topology는 같다. Donor URDF에는 이미
`floating_base_joint`가 있으므로 별도 free-flyer를 추가하면 안 된다. 두 asset은 Unitree
`g1_29dof_rev_1_0` 계열로 연결되지만 audit 재현에는 위 revision과 hash를 고정한다.

사용자가 제시한 `unitree_ros/robots/g1_d_description/g1_d.urdf`는 이 비교 대상이 아니다. 이것은 `G1_D`라는
AGV/wheel/lift 기반 상체 model로, 29-DoF 보행 G1의 hip/knee/ankle tree가 없다. 현재 official family를 새로
받아 쓰는 것보다 paper donor의 pinned URDF를 oracle로 유지한다.

## Loader와 model-surface 감사

Pinocchio 4.1.0에서 native URDF/MJCF model loader와 geometry loader는 모두 동작했고, 두 asset이 참조하는 35개
mesh가 모두 해석됐다. 두 model의 `(collision, visual)` geometry object 수는 `(36,35)`다. cpin cast 후 두 model
모두 `(nq,nv,njoints)=(36,35,31)`이다.

Raw model 차이는 다음과 같다.

- MJCF는 29개 joint에 `armature=0.01`, `friction=0.3`을 둔다. URDF는 둘 다 0이다.
- URDF에는 29개 finite velocity limit가 있지만 MJCF parsed model에는 없다. Free-base position bound도 다르다.
- Effort limit와 actuated joint order는 같다. Joint position limit 최대차는 `2.054e-6 rad`이다.
- Total mass 차이는 `2.0e-8 kg`이지만 fixed-link aggregation 때문에 inertia entry 최대차는 `5.2496e-5`다.
- Donor의 8개 foot-corner와 2개 hand contact frame은 MJCF에 named frame으로 존재하지 않는다. MJCF의 8개 foot
  sphere 좌표는 donor corner 좌표와 정확히 같으므로 native preprocessing에서 frame으로 합성할 수 있다.
- MuJoCo free-joint quaternion은 `wxyz`, Pinocchio configuration은 `xyzw`다. 명시적 변환 없이 state를 비교하면
  안 된다.
- Menagerie actuator는 position actuator다. NLP에서 복원한 torque를 MuJoCo `ctrl`에 그대로 넣는 것은 같은 plant
  action semantics가 아니다.

## 수치 결과

환경은 Python 3.13.15, Pinocchio/cpin 4.1.0, CasADi 3.8.0, MuJoCo 3.11.0이다. Seed 41의 deterministic 5-state
corpus를 사용했다.

### Loader 내부 parity

| 비교 | 최대 절대차 | 판정 |
| --- | ---: | --- |
| Wiki G1 URDF native-cpin regression | `<=2e-10` | 5 tests 통과 |
| Generic cpin API probe, RNEA/derivatives | `9.237e-14` | `2e-10` gate 통과 |
| MJCF native-cpin RNEA | `3.553e-15` | 통과 |
| MJCF native-cpin frame position | `2.776e-17` | 통과 |
| MuJoCo-Pinocchio same-MJCF mass matrix, `v=0` | `2.665e-15` | engine/parser parity 통과 |
| MuJoCo-Pinocchio same-MJCF bias, `v=0` | `5.684e-14` | engine/parser parity 통과 |

### URDF-MJCF cross-model parity

| quantity | raw 최대 절대차 | normalized 최대 절대차 |
| --- | ---: | ---: |
| joint world position | `3.004e-7 m` | unchanged |
| joint world rotation | `1.456e-6 rad` | unchanged |
| contact world position | missing named frames | `3.046e-7 m` |
| contact LWA Jacobian | missing named frames | `1.456e-6` |
| mass matrix | `1.0002343e-2` | `4.367883e-5` |
| nonlinear effects | `5.059256e-5` | `5.059256e-5` |
| RNEA | `1.065770e-2` | `4.683247e-5` |
| COM | `5.303e-8 m` | unchanged |

Normalization은 permanent asset edit가 아니라 native MJCF model의 `armature`, `rotorInertia`, `friction`,
`damping`, position/velocity/effort limit를 URDF profile로 복사하고 contact frame 10개를 합성한 one-shot probe다.
남은 차이는 주로 rounded inertial values와 fixed-link aggregation에서 온다.

## 성공 기준

이 audit 이후 model adoption 전 다음 기준을 **관측값과 독립적으로** 고정해야 한다.

1. `nq/nv`, actuated joint 이름·순서·parent와 action semantics가 exact match여야 한다.
2. Dynamics/limit/contact profile의 차이는 asset conversion으로 제거하거나 명시적인 waiver를 가져야 한다.
3. 각 accepted model의 native-cpin primitive는 현재 wiki gate인 `atol=rtol=2e-10`을 통과해야 한다.
4. Cross-model spatial/dynamics tolerance는 아직 미지정이다. 위 normalized 수치를 사후 threshold로 삼지 않는다.
5. 원 artifact를 확보하거나 같은 pinned environment에서 재생성한 뒤, node/interval canonical residual, objective,
   terminal state와 MuJoCo state playback을 함께 비교해야 trajectory parity를 주장할 수 있다.

따라서 현재 **raw model 교체 주장은 기각**, normalized model은 **후보**, 원 SE3_TrajOpt trajectory와의 동일성은
**missing artifact로 blocked**다.

## 2026-08-26 snapshot 검사

아래 명령과 `5 tests` 결과는 `research-wiki@c79accb860e135d0e4bcf7560409d65af6669e71` 및 historical
`mj_rl` diagnostic environment에 고정된 기록이다. 현재 dirty worktree의 test 수나 active `mj_opt` Function
preflight와 동일한 상태로 해석하지 않는다.

```bash
cd /home/frlab/research-wiki
PYTHONDONTWRITEBYTECODE=1 /home/frlab/anaconda3/envs/mj_rl/bin/python \
  -m unittest tests.test_casadi_trajectory_optimization.TestSE3TrajOptCpinCoverage -v

/home/frlab/anaconda3/envs/mj_rl/bin/python \
  /home/frlab/research-wiki/.agents/skills/casadi-pinocchio/scripts/probe_cpin_api.py
```

고정 revision에서 첫 명령은 5 tests를 통과했다. Donor optimizer 자체는 당시 environment에 `cyipopt`, `example_robot_data`,
`meshcat`가 없고 saved trajectory도 없어 실행하지 않았다. Dependency 설치나 donor file 수정은 이 audit 범위에
포함하지 않았다.
