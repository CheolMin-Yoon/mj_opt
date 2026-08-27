from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

G1_URDF = Path(
    "/home/frlab/reference/se3_trajopt/src/robots/g1/g1/urdf/g1_29dof.urdf"
)
CONTACT_FRAME = "left_foot_upper_left"
ATOL = 2.0e-10
SEED = 41
SAMPLES = 5


def _make_symbolic_external_forces(
    cmodel: cpin.Model,
    frame_id: int,
    force_local: ca.SX,
) -> cpin.StdVec_Force:
    frame = cmodel.frames[frame_id]

    force_at_parent_joint = frame.placement.act(
        cpin.Force(force_local, ca.SX.zeros(3, 1))
    )

    external_forces = cpin.StdVec_Force()
    for joint_id in range(cmodel.njoints):
        external_forces.append(
            force_at_parent_joint
            if joint_id == frame.parentJoint
            else cpin.Force.Zero()
        )

    return external_forces


def _make_native_external_forces(
    model: pin.Model,
    frame_id: int,
    force_local: np.ndarray,
) -> list[pin.Force]:
    frame = model.frames[frame_id]

    external_forces = [
        pin.Force.Zero()
        for _ in range(model.njoints)
    ]

    external_forces[frame.parentJoint] = frame.placement.act(
        pin.Force(force_local, np.zeros(3))
    )

    return external_forces


def _make_rnea_function(
    model: pin.Model,
    frame_id: int,
) -> ca.Function:
    cmodel = cpin.Model(model)

    q = ca.SX.sym("q", model.nq)
    v = ca.SX.sym("v", model.nv)
    a = ca.SX.sym("a", model.nv)
    force_local = ca.SX.sym("force_local", 3)

    external_forces = _make_symbolic_external_forces(
        cmodel,
        frame_id,
        force_local,
    )

    value_data = cmodel.createData()
    tau = cpin.rnea(
        cmodel,
        value_data,
        q,
        v,
        a,
        external_forces,
    )

    derivative_data = cmodel.createData()
    cpin.computeRNEADerivatives(
        cmodel,
        derivative_data,
        q,
        v,
        a,
        external_forces,
    )

    return ca.Function(
        "g1_rnea_probe",
        [q, v, a, force_local],
        [
            tau,
            derivative_data.dtau_dq,
            derivative_data.dtau_dv,
            derivative_data.M,
        ],
        ["q", "v", "a", "force_local"],
        [
            "tau",
            "dtau_dq",
            "dtau_dv",
            "mass_matrix",
        ],
    ).expand()


def test_g1_rnea_and_derivatives_native_cpin() -> None:
    assert G1_URDF.is_file()

    model = pin.buildModelFromUrdf(str(G1_URDF))
    assert (model.nq, model.nv) == (36, 35)

    frame_id = model.getFrameId(CONTACT_FRAME)
    assert frame_id < model.nframes

    rnea_function = _make_rnea_function(
        model,
        frame_id,
    )

    random = np.random.default_rng(SEED)
    neutral = pin.neutral(model)

    for _ in range(SAMPLES):
        tangent = random.normal(
            scale=0.025,
            size=model.nv,
        )
        q = pin.integrate(
            model,
            neutral,
            tangent,
        )
        v = random.normal(
            scale=0.2,
            size=model.nv,
        )
        a = random.normal(
            scale=0.3,
            size=model.nv,
        )
        force_local = random.normal(
            scale=15.0,
            size=3,
        )

        external_forces = _make_native_external_forces(
            model,
            frame_id,
            force_local,
        )

        tau_native = pin.rnea(
            model,
            model.createData(),
            q,
            v,
            a,
            external_forces,
        )

        derivative_data = model.createData()
        pin.computeRNEADerivatives(
            model,
            derivative_data,
            q,
            v,
            a,
            external_forces,
        )

        result = rnea_function(
            q=q,
            v=v,
            a=a,
            force_local=force_local,
        )

        np.testing.assert_allclose(
            np.asarray(result["tau"]).reshape(-1),
            tau_native,
            atol=ATOL,
            rtol=ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(result["dtau_dq"]),
            derivative_data.dtau_dq,
            atol=ATOL,
            rtol=ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(result["dtau_dv"]),
            derivative_data.dtau_dv,
            atol=ATOL,
            rtol=ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(result["mass_matrix"]),
            derivative_data.M,
            atol=ATOL,
            rtol=ATOL,
        )


def test_g1_contact_force_frame_and_rnea_sign() -> None:
    assert G1_URDF.is_file()

    model = pin.buildModelFromUrdf(str(G1_URDF))

    frame_id = model.getFrameId(CONTACT_FRAME)
    assert frame_id < model.nframes

    tangent = np.zeros(model.nv)
    tangent[3:6] = [0.2, -0.15, 0.3]
    tangent[6:] = np.linspace(
        -0.1,
        0.1,
        model.nv - 6,
    )

    q = pin.integrate(
        model,
        pin.neutral(model),
        tangent,
    )
    v = np.linspace(
        -0.2,
        0.2,
        model.nv,
    )
    a = np.linspace(
        0.1,
        -0.08,
        model.nv,
    )

    data = model.createData()
    pin.forwardKinematics(
        model,
        data,
        q,
        v,
        a,
    )
    pin.updateFramePlacements(
        model,
        data,
    )

    jacobian_local = pin.computeFrameJacobian(
        model,
        data,
        q,
        frame_id,
        pin.LOCAL,
    )[:3, :]

    jacobian_world = pin.computeFrameJacobian(
        model,
        data,
        q,
        frame_id,
        pin.LOCAL_WORLD_ALIGNED,
    )[:3, :]

    force_local = np.array([
        12.0,
        -3.0,
        100.0,
    ])
    force_world = (
        data.oMf[frame_id].rotation
        @ force_local
    )

    np.testing.assert_allclose(
        jacobian_local.T @ force_local,
        jacobian_world.T @ force_world,
        atol=ATOL,
        rtol=ATOL,
    )

    external_forces = _make_native_external_forces(
        model,
        frame_id,
        force_local,
    )

    tau_native = pin.rnea(
        model,
        model.createData(),
        q,
        v,
        a,
        external_forces,
    )

    mass_matrix = pin.crba(
        model,
        model.createData(),
        q,
    )
    bias = pin.nonLinearEffects(
        model,
        model.createData(),
        q,
        v,
    )

    tau_from_world_force = (
        mass_matrix @ a
        + bias
        - jacobian_world.T @ force_world
    )

    np.testing.assert_allclose(
        tau_from_world_force,
        tau_native,
        atol=ATOL,
        rtol=ATOL,
    )
