import sys
import numpy as np
import mujoco
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

np.set_printoptions(precision=4, suppress=True)

MJCF = '/home/frlab/mj_opt/xmls/systems/universal_robots_ur5e/scene_torque.xml'
URDF = '/home/frlab/mj_opt/xmls/robots/ur_description/urdf/ur5e.urdf'
PKG  = '/home/frlab/mj_opt/xmls/robots'

print("="*60)
print("🚀 [START] MuJoCo vs Pinocchio 정합성 테스트 (Fixed Base)")
print("="*60)

# ---------------------------------------------------------
# [STEP 1] 모델 초기화 및 환경 동기화
# ---------------------------------------------------------
print("\n▶ [STEP 1] 모델 초기화 및 환경 동기화")

# MuJoCo
mj_model = mujoco.MjModel.from_xml_path(MJCF)
mj_data  = mujoco.MjData(mj_model)

# Pinocchio (fixed base: root_joint=None)
robot  = RobotWrapper.BuildFromURDF(URDF, package_dirs=[PKG], root_joint=None)
model  = robot.model
data   = model.createData()

# 중력 동기화
model.gravity.linear = mj_model.opt.gravity

# Armature 동기화 (MJ → Pin)
model.armature[:] = mj_model.dof_armature

# passive force 제거 (fair 비교)
mj_model.dof_damping[:]      = 0.0
mj_model.dof_frictionloss[:] = 0.0

# 관절 순서 확인
pin_joint_names = list(model.names[1:])   # universe 제외
mj_joint_names  = [mj_model.joint(i).name for i in range(mj_model.njnt)]
assert pin_joint_names == mj_joint_names, \
    f"관절 순서 불일치!\n  Pin: {pin_joint_names}\n  MJ:  {mj_joint_names}"
print("✅ 초기화 완료, 관절 순서 일치")
print(f"   nq={model.nq}, nv={model.nv}, nu={mj_model.nu}")
print(f"   joints: {pin_joint_names}")

# ---------------------------------------------------------
# [STEP 2] 기구학(Kinematics) 검증 — EE 위치 비교
# ---------------------------------------------------------
print("\n▶ [STEP 2] 기구학(Kinematics) 검증")

q_home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.])
mj_data.qpos[:] = q_home
mujoco.mj_forward(mj_model, mj_data)

pin.forwardKinematics(model, data, q_home)
pin.updateFramePlacements(model, data)

ee_id_pin = model.getFrameId("tool0")
ee_pos_pin = data.oMf[ee_id_pin].translation

ee_id_mj  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
ee_pos_mj = mj_data.site_xpos[ee_id_mj]

ee_err = np.linalg.norm(ee_pos_pin - ee_pos_mj)
print(f"  EE 위치 오차: {ee_err:.2e} {'✅' if ee_err < 1e-4 else '🚨'}")
print(f"  Pin EE: {ee_pos_pin.round(4)}")
print(f"  MJ  EE: {ee_pos_mj.round(4)}")

# ---------------------------------------------------------
# [STEP 3] 자코비안(Jacobian) 검증
# ---------------------------------------------------------
print("\n▶ [STEP 3] 자코비안(Jacobian) 검증")

pin.computeAllTerms(model, data, q_home, np.zeros(model.nv))
pin.updateFramePlacements(model, data)
pin.computeJointJacobians(model, data, q_home)

J_pin = pin.getFrameJacobian(model, data, ee_id_pin, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

J_mj = np.zeros((6, mj_model.nv))
mujoco.mj_jacSite(mj_model, mj_data, J_mj[:3], J_mj[3:], ee_id_mj)

jac_err = np.max(np.abs(J_pin - J_mj))
print(f"  자코비안 오차: {jac_err:.2e} {'✅' if jac_err < 1e-4 else '🚨'}")

# ---------------------------------------------------------
# [STEP 4] 동역학(Dynamics) 검증
# ---------------------------------------------------------
print("\n▶ [STEP 4] 동역학(Dynamics) 검증")

rng = np.random.default_rng(seed=42)
q_rand  = rng.uniform(-np.pi, np.pi, model.nq)
dq_rand = rng.standard_normal(model.nv) * 0.5

# 양 엔진 동기화
mj_data.qpos[:] = q_rand
mj_data.qvel[:] = dq_rand
mujoco.mj_forward(mj_model, mj_data)

pin.computeAllTerms(model, data, q_rand, dq_rand)
pin.updateFramePlacements(model, data)
pin.computeJointJacobiansTimeVariation(model, data, q_rand, dq_rand)

# ── Mass matrix ──────────────────────────────────────────
M_pin_raw = data.M.copy()
M_pin = M_pin_raw + M_pin_raw.T - np.diag(np.diag(M_pin_raw))  # symmetrize

M_mj = np.zeros((mj_model.nv, mj_model.nv))
mujoco.mj_fullM(mj_model, M_mj, mj_data.qM)

diag_err     = np.max(np.abs(np.diag(M_pin) - np.diag(M_mj)))
off_diag_err = np.max(np.abs((M_pin - M_mj) - np.diag(np.diag(M_pin - M_mj))))
print(f"  질량 행렬 대각 오차  : {diag_err:.6e} {'✅' if diag_err < 1e-4 else '🚨'}")
print(f"  질량 행렬 비대각 오차: {off_diag_err:.6e} {'✅' if off_diag_err < 1e-4 else '🚨'}")

# ── Gravity ──────────────────────────────────────────────
g_pin = data.g.copy()

mj_data.qvel[:] = 0; mujoco.mj_forward(mj_model, mj_data)
g_mj = mj_data.qfrc_bias.copy()
mj_data.qvel[:] = dq_rand; mujoco.mj_forward(mj_model, mj_data)

gravity_err = np.max(np.abs(g_pin - g_mj))
print(f"  Gravity 오차         : {gravity_err:.6e} {'✅' if gravity_err < 1e-4 else '🚨'}")

# ── 방법 1: NLE 전체 비교 ────────────────────────────────
nle_pin = data.nle.copy()
nle_mj  = mj_data.qfrc_bias.copy()
nle_err = np.max(np.abs(nle_pin - nle_mj))
print(f"\n  [방법 1] NLE 전체 오차 (g+C@dq): {nle_err:.6e} {'✅' if nle_err < 1e-4 else '🚨'}")

# ── 방법 2: RNEA vs mj_inverse ───────────────────────────
# fixed base: qacc=0 → tau = NLE, frame 변환 없음
tau_rnea = pin.rnea(model, data, q_rand, dq_rand, np.zeros(model.nv))

mj_data.qacc[:] = 0.0
mujoco.mj_inverse(mj_model, mj_data)
tau_mj_inv = mj_data.qfrc_inverse.copy()

rnea_err = np.max(np.abs(tau_rnea - tau_mj_inv))
print(f"  [방법 2] RNEA vs mj_inverse 오차: {rnea_err:.6e} {'✅' if rnea_err < 1e-4 else '🚨'}")

# ── 방법 3: ABA vs mj_forward (tau=0) ────────────────────
# fixed base: frame 변환 없음 → fair 비교
tau_zero = np.zeros(model.nv)
ddq_pin  = pin.aba(model, data, q_rand, dq_rand, tau_zero)

mj_data.qvel[:] = dq_rand
mj_data.ctrl[:] = 0.0
mujoco.mj_forward(mj_model, mj_data)
ddq_mj = mj_data.qacc.copy()

ddq_err = np.max(np.abs(ddq_pin - ddq_mj))
print(f"  [방법 3] ABA vs mj_forward 오차 (tau=0): {ddq_err:.6e} {'✅' if ddq_err < 1e-2 else '🚨'}")

# ---------------------------------------------------------
# 최종 요약
# ---------------------------------------------------------
print("\n" + "="*60)
all_pass = all(e < 1e-4 for e in [ee_err, jac_err, diag_err, off_diag_err,
                                    gravity_err, nle_err, rnea_err])
if all_pass:
    print("🎉 [SUCCESS] 모든 항목 통과! 두 엔진이 동일한 물리로 동작합니다.")
else:
    print("⚠️  [WARNING] 일부 항목 오차 있음. 위 결과 확인.")
print("="*60)

# ── 해석 노트 ─────────────────────────────────────────────
# [Fixed base 비교의 의미]
# Fixed base는 floating base와 달리 frame 변환(world↔body)이 없어서
# 방법 1(NLE), 방법 2(RNEA vs mj_inverse), 방법 3(ABA vs mj_forward) 모두
# frame 불일치 없이 공정하게 비교 가능하다.
#
# [방법 2가 가장 엄격한 비교]
# RNEA(ddq=0) == NLE이므로 방법 1과 동치이나, mj_inverse를 통해
# MuJoCo 역동역학 API를 직접 대조한다는 점에서 더 신뢰도 높다.
#
# [방법 3의 한계]
# tau=0에서 mj_forward는 contact constraint solver까지 포함하므로
# 로봇이 공중에 떠있지 않으면 바닥 충돌 반력이 qacc에 섞인다.
# 따라서 방법 3은 참고용이며 방법 1, 2가 주 비교 지표다.
