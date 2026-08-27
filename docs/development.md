# Development and validation

## Source-layout rule

`source/` 내부 파일·폴더 이름은 사용자가 정합니다. Agent는 구체적인 구현 요청 없이 module scaffold, generic
base class, registry 또는 placeholder solver를 만들지 않습니다. 이 문서는 구현 순서와 gate만 고정합니다.

## Implementation order

1. Model/source revision, `(nq,nv)`, joint/frame/contact order와 dependency probe를 고정한다.
2. Native Pinocchio와 accepted CasADi Function의 value/derivative/frame parity를 통과한다.
3. 한 interval의 parameterization decode, transition, dynamics와 contact truth cases를 통과한다.
4. `N/N+1` packing, row schema, scaling과 intentionally infeasible problem을 검사한다.
5. Deterministic G1 solve 뒤 canonical residual, dense audit와 artifact round-trip을 통과한다.
6. Accepted artifact만 MuJoCo state playback, torque sanity와 controller tracking에 넘긴다.

매 단계는 첫 미통과 gate에서 멈춥니다. Tiny NLP나 Function construction을 G1 trajectory solve로, node residual을
dense feasibility로, MuJoCo animation을 tracking acceptance로 확대하지 않습니다.

## Environment boundary

Accepted base solver environment는 `/home/frlab/anaconda3/envs/mj_opt`이며 최소 Conda 입력은
`docs/config/environment.yml`이 소유합니다. CasADi는 공식 3.8.0 wheel `83b3cec`(SHA-256
`b58ae6f3784b5553461d70de9848ca5c893e5ae3eb81d00959dce2ce8484da58`), IPOPT는 wheel 내
`3.14.19.mod`, Pinocchio/cpin은 공식 `v4.1.0@2ae77666` source build입니다. Conda `casadi`, `pinocchio`,
`pin`, `ipopt` package는 추가하지 않습니다.

Pinocchio는 최종 prefix에서 Python/cpin을 함께 빌드합니다. 공식 wheel의 `libcasadi.so`가
`site-packages/casadi`에 있으므로 설치 뒤 cpin extension의 RPATH는
`$ORIGIN/../../../../lib:$ORIGIN/../casadi`, `libpinocchio_casadi.so`는
`$ORIGIN:$ORIGIN/python3.13/site-packages/casadi`로 고정합니다. Install file 목록은
`$CONDA_PREFIX/conda-meta/pinocchio-4.1.0-source-install-manifest.txt`에 둡니다.

Acceptance는 Python 3.13.15, CasADi revision/plugin, IPOPT tiny solve raw status, `pip check`, native/cpin API와
`2e-10` parity, `ldd`의 `not found` 0건 및 `mj_rl`/rollback/`codegen_env` RPATH 0건을 요구합니다.
Solver/compiler prefix와 `/home/frlab/mj_rl` training prefix는 분리하며 한쪽의 NumPy/Python/native dependency
문제를 다른 쪽에 설치해 해결하지 않습니다.

MuJoCo playback은 artifact consumer이므로 solver ABI와 같은 environment일 필요가 없습니다. 대신 model hash,
MuJoCo version, integrator/solver option, timestep, joint/actuator order와 applied torque trace를 고정합니다.

Repository-local tool과 problem configuration은 `docs/config/`에 둡니다. 설명만 있는 placeholder profile은
만들지 않고, 실제 consumer가 생길 때 model source/hash, `(nq,nv)`, joint/frame/contact order,
parameterization, contact schedule, boundary target, scaling과 solver option을 config에 기록합니다. URDF, MJCF와
mesh는 config 폴더로 복사하지 않습니다.

## Current preflight

```bash
PYTHONDONTWRITEBYTECODE=1 /home/frlab/anaconda3/envs/mj_opt/bin/python \
  -m pytest -p no:cacheprovider -q tests
PYTHONDONTWRITEBYTECODE=1 /home/frlab/anaconda3/envs/mj_opt/bin/python \
  -m compileall -q source
/home/frlab/anaconda3/envs/mj_opt/bin/python \
  -m ruff check --config docs/config/pyproject.toml .
```

현재 `source/`는 비어 있고 `tests/`는 exploratory native/cpin primitive probe만 포함합니다. 위 test가 통과해도
versioned project-consumed Function ABI나 optimizer 기능을 승인하지 않습니다. 구현 명령, test와 tolerance는 실제
source가 생길 때 사용자가 정한 API와 focused regression에 맞춰 추가합니다.
