from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

G1_URDF = Path(
    "/home/frlab/reference/se3_trajopt/src/robots/g1/g1/urdf/g1_29dof.urdf"
)
ATOL = 2.0e-10

# Q1. ATOL?이 뭐야
# A1. absolute tolerance, 즉 절대 허용 오차

# Q2. assert 문법이 뭐야
# A2. assert + 조건으로 True이면 다음줄로 진행

# Q3. delta로만 정의하면 이게 오일러 이산화? ZOH 안해도 되나
# A3. delta는 시간과 무관한 configuration tangent displacement, 즉 구성 다양체 위의 작은 이동량이다.


def test_g1_integrate_difference_native_cpin() -> None:
    assert G1_URDF.is_file()

    model = pin.buildModelFromUrdf(str(G1_URDF))
    assert (model.nq, model.nv) == (36, 35)

    cmodel = cpin.Model(model)

    q = ca.SX.sym("q", model.nq)
    delta = ca.SX.sym("delta", model.nv)

    q_next = cpin.integrate(cmodel, q, delta)
    delta_recovered = cpin.difference(cmodel, q, q_next) # 이게 아마 remained인가

    manifold = ca.Function("g1_manifold_probe", [q, delta], [q_next, delta_recovered], ["q", "delta"], ["q_next", "delta_recovered"]).expand()
    # expand()가 뭐야


    q0 = pin.neutral(model)
    # 이거 dt임? -0.03 0.03이 뭐야 섭동?
    delta0 = np.linspace(-0.03, 0.03, model.nv)


    q1_native = pin.integrate(model, q0, delta0)

    result = manifold(q=q0, delta=delta0)
    q1_cpin = np.asarray(result["q_next"]).reshape(-1)
    delta_cpin = np.asarray(result["delta_recovered"]).reshape(-1)

    np.testing.assert_allclose(
        q1_cpin,
        q1_native,
        atol=ATOL,
        rtol=ATOL,
    )
    np.testing.assert_allclose(
        delta_cpin,
        delta0,
        atol=ATOL,
        rtol=ATOL,
    )

    assert manifold.size_in("q") == (36, 1)
    assert manifold.size_in("delta") == (35, 1)
    assert manifold.size_out("q_next") == (36, 1)
    assert manifold.size_out("delta_recovered") == (35, 1)

    print("nq/nv:", model.nq, model.nv)
    print(
        "native-cpin max error:",
        np.max(np.abs(q1_cpin - q1_native)),
    )
    print(
        "local inverse max error:",
        np.max(np.abs(delta_cpin - delta0)),
    )
    print(
        "graph nodes/instructions:",
        manifold.n_nodes(),
        manifold.n_instructions(),
    )
    print(
        "workspace iw/w:",
        manifold.sz_iw(),
        manifold.sz_w(),
    )
