"""G1 SE(3)-tangent 표현의 한 구간 semi-implicit transition Function."""

from __future__ import annotations

import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
from trajectory_optimization.manifold_state import ManifoldStateTransition

G1_NQ = 36
G1_NV = 35
G1_NJOINTS = 31

FUNCTION_NAME = "eval_g1_se3_semi_implicit_transition_v1"


def _validate_g1_model(model: pin.Model) -> None:
    """Function ABI가 전제하는 G1 model 구조를 확인한다."""

    dimensions = (
        int(model.nq),
        int(model.nv),
        int(model.njoints),
    )
    expected_dimensions = (
        G1_NQ,
        G1_NV,
        G1_NJOINTS,
    )

    if dimensions != expected_dimensions:
        raise ValueError(
            "Expected G1 dimensions "
            f"{expected_dimensions}, got {dimensions}"
        )

    floating_base = model.joints[1]
    floating_base_layout = (
        int(floating_base.idx_q),
        int(floating_base.nq),
        int(floating_base.idx_v),
        int(floating_base.nv),
    )
    expected_floating_base_layout = (
        0,
        7,
        0,
        6,
    )

    if floating_base_layout != expected_floating_base_layout:
        raise ValueError(
            "Expected the donor URDF floating base layout "
            f"{expected_floating_base_layout}, "
            f"got {floating_base_layout}"
        )

    actuated_joint_layout = tuple(
        (
            int(model.joints[joint_id].nq),
            int(model.joints[joint_id].nv),
        )
        for joint_id in range(2, model.njoints)
    )

    if actuated_joint_layout != ((1, 1),) * 29:
        raise ValueError(
            "Expected 29 scalar actuated joints after the floating base"
        )


def decode_g1_se3_tangent(
    configuration_tangent: ca.SX,
) -> ca.SX:
    """SE(3) log coordinate와 joint 좌표를 Pinocchio q로 변환한다.

    Input
    -----
    configuration_tangent:
        [base_se3_log(6), actuated_joint_position(29)], shape (35, 1)

    Output
    ------
    Pinocchio configuration:
        [base_translation(3), base_quaternion_xyzw(4), joints(29)],
        shape (36, 1)
    """

    if configuration_tangent.shape != (G1_NV, 1):
        raise ValueError(
            "configuration_tangent must have shape "
            f"({G1_NV}, 1), got {configuration_tangent.shape}"
        )

    floating_base_configuration = cpin.exp6_quat(
        configuration_tangent[:6]
    )
    actuated_joint_configuration = configuration_tangent[6:]

    return ca.vertcat(
        floating_base_configuration,
        actuated_joint_configuration,
    )


def make_g1_se3_semi_implicit_transition_function(
    model: pin.Model,
) -> ca.Function:
    """Donor-compatible G1 one-interval transition Function을 만든다.

    ``time_step``은 Function parameter이며 아직 NLP decision이 아니다.

    Donor defect convention:

        difference(next_configuration, predicted_configuration)

    이 순서는 shared ManifoldStateTransition의 기본 defect 순서와
    반대이므로 prediction만 재사용하고 defect는 여기서 조합한다.
    """

    _validate_g1_model(model)

    transition = ManifoldStateTransition(
        model,
        integration_scheme=(
            ManifoldStateTransition.SEMI_IMPLICIT_EULER
        ),
    )

    configuration_tangent = ca.SX.sym(
        "configuration_tangent",
        G1_NV,
    )
    generalized_velocity = ca.SX.sym(
        "generalized_velocity",
        G1_NV,
    )
    generalized_acceleration = ca.SX.sym(
        "generalized_acceleration",
        G1_NV,
    )
    time_step = ca.SX.sym("time_step")
    next_configuration_tangent = ca.SX.sym(
        "next_configuration_tangent",
        G1_NV,
    )
    next_generalized_velocity = ca.SX.sym(
        "next_generalized_velocity",
        G1_NV,
    )

    configuration = decode_g1_se3_tangent(
        configuration_tangent
    )
    next_configuration = decode_g1_se3_tangent(
        next_configuration_tangent
    )

    (
        configuration_prediction,
        generalized_velocity_prediction,
    ) = transition.predict(
        configuration,
        generalized_velocity,
        generalized_acceleration,
        time_step,
    )

    # SE3_TrajOpt donor order:
    # difference(q_next, q_prediction)
    configuration_defect = cpin.difference(
        transition.model,
        next_configuration,
        configuration_prediction,
    )

    generalized_velocity_defect = (
        next_generalized_velocity
        - generalized_velocity_prediction
    )

    return ca.Function(
        FUNCTION_NAME,
        [
            configuration_tangent,
            generalized_velocity,
            generalized_acceleration,
            time_step,
            next_configuration_tangent,
            next_generalized_velocity,
        ],
        [
            configuration_defect,
            generalized_velocity_defect,
        ],
        [
            "configuration_tangent",
            "generalized_velocity",
            "generalized_acceleration",
            "time_step",
            "next_configuration_tangent",
            "next_generalized_velocity",
        ],
        [
            "configuration_defect",
            "generalized_velocity_defect",
        ],
    )


__all__ = [
    "FUNCTION_NAME",
    "G1_NQ",
    "G1_NV",
    "decode_g1_se3_tangent",
    "make_g1_se3_semi_implicit_transition_function",
]
