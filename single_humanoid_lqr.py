#!/usr/bin/python

import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import scipy.linalg

def lqr(model, data, qpos0, ctrl0):    
    nu = model.nu  # Alias for the number of actuators.
    R_mat = np.eye(nu)

    nv = model.nv  # Shortcut for the number of DoFs.

    # Get the Jacobian for the root body (torso) CoM.
    mj.mj_resetData(model, data)
    data.qpos = qpos0
    mj.mj_forward(model, data)
    jac_com = np.zeros((3, nv))
    torso_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'torso')
    mj.mj_jacSubtreeCom(model, data, jac_com, torso_body_id)

    # Get the Jacobian for the left foot.
    jac_foot = np.zeros((3, nv))
    foot_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'foot_left')
    mj.mj_jacBodyCom(model, data, jac_foot, None, foot_body_id)

    jac_diff = jac_com - jac_foot
    Qbalance = jac_diff.T @ jac_diff

    # Get all joint names.
    joint_names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]

    # Get indices into relevant sets of joints.
    root_dofs = range(6)
    body_dofs = range(6, nv)
    abdomen_dofs = [
        model.joint(name).dofadr[0]
        for name in joint_names
        if 'abdomen' in name
        and not 'z' in name
    ]
    left_leg_dofs = [
        model.joint(name).dofadr[0]
        for name in joint_names
        if 'left' in name
        and ('hip' in name or 'knee' in name or 'ankle' in name)
        and not 'z' in name
    ]
    balance_dofs = abdomen_dofs + left_leg_dofs
    other_dofs = np.setdiff1d(body_dofs, balance_dofs)

    # Cost coefficients.
    BALANCE_COST        = 1000  # Balancing.
    BALANCE_JOINT_COST  = 3     # Joints required for balancing.
    OTHER_JOINT_COST    = .3    # Other joints.

    # Construct the Qjoint matrix.
    Qjoint = np.eye(nv)
    Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
    Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
    Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

    # Construct the Q matrix for position DoFs.
    Qpos = BALANCE_COST * Qbalance + Qjoint

    # Qpos = np.eye(nv)

    # No explicit penalty for velocities.
    Q = np.block([[Qpos, np.zeros((nv, nv))],
                [np.zeros((nv, 2*nv))]])


    # Set the initial state and control.
    mj.mj_resetData(model, data)
    data.ctrl = ctrl0
    data.qpos = qpos0

    # Allocate the A and B matrices, compute them.
    A = np.zeros((2*nv, 2*nv))
    B = np.zeros((2*nv, nu))
    epsilon = 1e-6
    centered = True
    mj.mjd_transitionFD(model, data, epsilon, centered, A, B, None, None)

    # Solve discrete Riccati equation.
    P = scipy.linalg.solve_discrete_are(A, B, Q, R_mat)

    # Compute the feedback gain matrix K.
    K = np.linalg.inv(R_mat + B.T @ P @ B) @ B.T @ P @ A

    data.qpos = qpos0

    # Allocate position difference dq.
    dq = np.zeros(model.nv)

    return K, dq, P, A, B, Q

def compute_stability_metrics(model, data, P, A, B, K, qpos0, prev_state_error=None, dt=0.0):
    """
    Compute comprehensive stability metrics for the LQR-controlled system
    """
    # Current state deviation from desired
    dq_current = np.zeros(model.nv)
    mj.mj_differentiatePos(model, dq_current, 1, qpos0, data.qpos)
    state_error = np.concatenate([dq_current, data.qvel])
    
    # 1. LQR Cost-to-Go (Primary stability metric)
    cost_to_go = state_error.T @ P @ state_error
    
    # 2. Normalized stability score (0-1 scale)
    stability_score = 1.0 / (1.0 + 0.1 * cost_to_go)
    
    # 3. Eigenvalue stability analysis
    A_cl = A - B @ K  # Closed-loop dynamics
    eigvals = np.linalg.eigvals(A_cl)
    
    # Stability margin - distance from instability
    real_parts = np.real(eigvals)
    stability_margin = np.min(-real_parts) if np.all(real_parts < 0) else 0
    
    # Damping ratios for oscillatory modes
    damping_ratios = []
    for eig in eigvals:
        if np.imag(eig) != 0:  # Complex pair
            wn = np.abs(eig)
            zeta = -np.real(eig) / wn
            damping_ratios.append(zeta)
    
    avg_damping = np.mean(damping_ratios) if damping_ratios else 1.0
    min_damping = np.min(damping_ratios) if damping_ratios else 1.0
    
    # 4. Physical stability metrics
    com_velocity = np.linalg.norm(data.cvel[1][3:6])  # Torso linear velocity
    com_height = data.subtree_com[1][2]  # Torso COM height
    
    # 5. Control effort
    torque_usage = np.linalg.norm(data.ctrl)
    max_torque = np.max(np.abs(data.ctrl))
    
    # 6. Lyapunov rate of change (if previous state available)
    lyapunov_rate = 0.0
    if prev_state_error is not None and dt > 0:
        V_prev = prev_state_error.T @ P @ prev_state_error
        V_current = cost_to_go
        lyapunov_rate = (V_current - V_prev) / dt
    
    # 7. Comprehensive combined score
    comprehensive_score = (
        stability_score * 0.4 +                    # LQR performance
        min_damping * 0.3 +                        # Worst-case damping
        np.exp(-com_velocity) * 0.2 +              # COM velocity penalty
        np.exp(-torque_usage/10) * 0.1             # Control effort penalty
    )
    
    return {
        'stability_score': stability_score,
        'comprehensive_score': comprehensive_score,
        'lqr_cost': cost_to_go,
        'stability_margin': stability_margin,
        'avg_damping': avg_damping,
        'min_damping': min_damping,
        'com_velocity': com_velocity,
        'com_height': com_height,
        'torque_usage': torque_usage,
        'max_torque': max_torque,
        'lyapunov_rate': lyapunov_rate,
        'state_error': state_error.copy(),
        'is_stable': stability_margin > 0 and lyapunov_rate <= 0
    }


class Biped:
    def __init__(self, xml_path):
        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.sim_end = 110.0

        self.arms = None  # "fixed"
        
        # Stability tracking
        self.stability_history = []
        self.prev_state_error = None
        self.stability_data = []

        # Initialize GLFW and visualization
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        self.window = glfw.create_window(1200, 900, "Biped LQR", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # Initialize visualization structures
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # Set default camera
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        
        # Set camera configuration
        self.cam.azimuth = 120.89
        self.cam.elevation = -15.81
        self.cam.distance = 8.0
        self.cam.lookat = np.array([0.0, 0.0, 2.0])

        # Install GLFW callbacks for default mouse interaction
        glfw.set_key_callback(self.window, self._keyboard)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move)
        glfw.set_mouse_button_callback(self.window, self._mouse_button)
        glfw.set_scroll_callback(self.window, self._scroll)

        # Mouse state variables
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

    def _keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            self.stability_history = []
            self.prev_state_error = None
            self.stability_data = []

    def _mouse_button(self, window, button, act, mods):
        self.button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        glfw.get_cursor_pos(window)

    def _mouse_move(self, window, xpos, ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        if (not self.button_left) and (not self.button_middle) and (not self.button_right):
            return

        width, height = glfw.get_window_size(window)

        PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

        if self.button_right:
            action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self.model, action, dx/height, dy/height, self.scene, self.cam)

    def _scroll(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 * yoffset, self.scene, self.cam)

    def reset(self):
        # Stand on 2 legs
        qpos0, ctrl0 = self.stand_two_legs()

        # Get the LQR Controller Value K and stability matrices
        K, dq, P, A, B, Q = lqr(self.model, self.data, qpos0, ctrl0)

        # Perturbations 
        CTRL_STD, perturb = self.noise()

        # Reset stability tracking
        self.stability_history = []
        self.prev_state_error = None
        self.stability_data = []

        return qpos0, ctrl0, K, dq, CTRL_STD, perturb, P, A, B

    def controller(self, dq, qpos0, ctrl0, K, CTRL_STD, perturb, step, P, A, B):
        """
        LQR controller with stability monitoring
        """
        # Get state difference dx.
        mj.mj_differentiatePos(self.model, dq, 1, qpos0, self.data.qpos)
        dx = np.hstack((dq, self.data.qvel)).T

        # LQR control law.
        self.data.ctrl = ctrl0 - K @ dx

        if self.arms == "passive":
            self.data.ctrl[15] = 0
            self.data.ctrl[16] = 0
            self.data.ctrl[17] = 0
            self.data.ctrl[18] = 0
            self.data.ctrl[19] = 0
            self.data.ctrl[20] = 0

        elif self.arms == "fixed":
            self.data.ctrl[15] = ctrl0[15]
            self.data.ctrl[16] = ctrl0[16]
            self.data.ctrl[17] = ctrl0[17]
            self.data.ctrl[18] = ctrl0[18]
            self.data.ctrl[19] = ctrl0[19]
            self.data.ctrl[20] = ctrl0[20]

        # Add perturbation, increment step.
        self.data.ctrl += CTRL_STD * perturb[step]
        
        # Compute stability metrics
        dt = self.model.opt.timestep
        stability_metrics = compute_stability_metrics(
            self.model, self.data, P, A, B, K, qpos0, self.prev_state_error, dt
        )
        
        # Store for next iteration
        self.prev_state_error = stability_metrics['state_error']
        
        # Add to history
        self.stability_history.append(stability_metrics['stability_score'])
        self.stability_data.append(stability_metrics)
        
        # Print stability info periodically
        if step % 100 == 0:
            self.print_stability_info(stability_metrics, step)
            
        return stability_metrics

    def print_stability_info(self, metrics, step):
        """Print current stability information"""
        print(f"Step {step}: "
              f"Stability: {metrics['stability_score']:.3f}, "
              f"Comp: {metrics['comprehensive_score']:.3f}, "
              f"LQR Cost: {metrics['lqr_cost']:.1f}, "
              f"COM Vel: {metrics['com_velocity']:.3f}, "
              f"Torque: {metrics['torque_usage']:.2f}, "
              f"Stable: {metrics['is_stable']}")

    def simulate(self):
        step = 0
        qpos0, ctrl0, K, dq, CTRL_STD, perturb, P, A, B = self.reset()
        
        while not glfw.window_should_close(self.window):
            simstart = self.data.time

            # Apply periodic external forces
            x = 5 * int(self.data.time / 5)
            y = x + 0.2
            # if self.data.time >= x and self.data.time <= y:
            #     self.apply_external_forces([55.0 + 2 * x / 10, 0.0, 0.0])
            # else:
            #     self.apply_external_forces([0.0, 0.0, 0.0])
            
            while (self.data.time - simstart < 1.0 / 60.0):
                # Step simulation environment
                mj.mj_step(self.model, self.data)

                # Apply control with stability monitoring
                stability_metrics = self.controller(dq, qpos0, ctrl0, K, CTRL_STD, perturb, step, P, A, B)
                step += 1
                if step >= 2390:
                    step = 0

            if self.data.time >= self.sim_end:
                # Print final stability summary
                self.print_final_stability_summary()
                break

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Show joint frames
            self.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = 0

            # Enable perturbation force visualisation.
            self.opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = 1

            # Enable contact force visualisation.
            self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

            # Set the scale of visualized contact forces to 1cm/N.
            self.model.vis.map.force = 0.01

            # Update scene and render
            self.cam.lookat[0] = self.data.qpos[0]
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()

    def print_final_stability_summary(self):
        """Print comprehensive stability analysis at the end of simulation"""
        if not self.stability_data:
            return
            
        print("\n" + "="*60)
        print("FINAL STABILITY ANALYSIS")
        print("="*60)
        
        # Calculate statistics
        stability_scores = [d['stability_score'] for d in self.stability_data]
        comp_scores = [d['comprehensive_score'] for d in self.stability_data]
        lqr_costs = [d['lqr_cost'] for d in self.stability_data]
        com_velocities = [d['com_velocity'] for d in self.stability_data]
        torque_usages = [d['torque_usage'] for d in self.stability_data]
        stable_states = [d['is_stable'] for d in self.stability_data]
        
        print(f"Average Stability Score: {np.mean(stability_scores):.3f} ± {np.std(stability_scores):.3f}")
        print(f"Average Comprehensive Score: {np.mean(comp_scores):.3f} ± {np.std(comp_scores):.3f}")
        print(f"Average LQR Cost: {np.mean(lqr_costs):.1f} ± {np.std(lqr_costs):.1f}")
        print(f"Average COM Velocity: {np.mean(com_velocities):.3f} ± {np.std(com_velocities):.3f}")
        print(f"Average Torque Usage: {np.mean(torque_usages):.2f} ± {np.std(torque_usages):.2f}")
        print(f"Stable States: {np.sum(stable_states)}/{len(stable_states)} ({np.mean(stable_states)*100:.1f}%)")
        print(f"Minimum Stability Score: {np.min(stability_scores):.3f}")
        print(f"Maximum Stability Score: {np.max(stability_scores):.3f}")
        
        # Stability classification
        avg_stability = np.mean(stability_scores)
        if avg_stability > 0.8:
            stability_class = "EXCELLENT"
        elif avg_stability > 0.6:
            stability_class = "GOOD"
        elif avg_stability > 0.4:
            stability_class = "MARGINAL"
        else:
            stability_class = "POOR"
            
        print(f"Overall Stability: {stability_class}")
        print("="*60)

    def quat2euler(self, quat):
        _quat = np.concatenate([quat[1:], quat[:1]])
        r = R.from_quat(_quat)
        euler = r.as_euler('xyz', degrees=False)
        return euler

    def stand_two_legs(self):
        mj.mj_resetDataKeyframe(self.model, self.data, 2)
        mj.mj_forward(self.model, self.data)
        self.data.qacc = 0
        mj.mj_inverse(self.model, self.data)

        height_offsets = np.linspace(-0.001, 0.001, 2001)
        vertical_forces = []
        for offset in height_offsets:
            mj.mj_resetDataKeyframe(self.model, self.data, 2)
            mj.mj_forward(self.model, self.data)
            self.data.qacc = 0
            self.data.qpos[2] += offset
            mj.mj_inverse(self.model, self.data)
            vertical_forces.append(self.data.qfrc_inverse[2])

        idx = np.argmin(np.abs(vertical_forces))
        best_offset = height_offsets[idx]

        mj.mj_resetDataKeyframe(self.model, self.data, 2)
        mj.mj_forward(self.model, self.data)
        self.data.qacc = 0
        self.data.qpos[2] += best_offset
        qpos0 = self.data.qpos.copy()
        mj.mj_inverse(self.model, self.data)
        qfrc0 = self.data.qfrc_inverse.copy()

        ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(self.data.actuator_moment)
        ctrl0 = ctrl0.flatten()

        mj.mj_resetData(self.model, self.data)
        self.data.qpos = qpos0
        self.data.ctrl = ctrl0

        return qpos0, ctrl0

    def noise(self):
        nu = self.model.nu

        DURATION = 30
        CTRL_STD = 0.05
        CTRL_RATE = 0.8

        np.random.seed(1)
        nsteps = int(np.ceil(DURATION / self.model.opt.timestep))
        perturb = np.random.randn(nsteps, nu)

        width = int(nsteps * CTRL_RATE / DURATION)
        kernel = np.exp(-0.5 * np.linspace(-3, 3, width)**2)
        kernel /= np.linalg.norm(kernel)
        for i in range(nu):
            perturb[:, i] = np.convolve(perturb[:, i], kernel, mode='same')

        return CTRL_STD, perturb

    def apply_external_forces(self, force):
        self.data.qfrc_applied[0] = force[0]
        self.data.qfrc_applied[1] = force[1]
        self.data.qfrc_applied[2] = force[2]

        bpos = self.data.xipos[1]
        ppos = bpos + force
        self.model.geom_size[20][1] = np.linalg.norm(force) / 400
        self.model.geom_rgba[20][3] = 1

        quat = np.zeros(4)
        mat = np.zeros(9)
        pertnorm = force / (np.linalg.norm(force) + 0.0001)
        mj.mju_quatZ2Vec(quat, pertnorm)
        mj.mju_quat2Mat(mat, quat)
        self.data.geom_xpos[20] = ppos
        self.data.geom_xmat[20] = mat

        if np.linalg.norm(np.array(force)) != 0:
            print("Force is applied:", force[0], self.data.qfrc_applied[0])

def main():
    xml_path = "models/humanoid.xml"
    sim = Biped(xml_path)
    sim.simulate()

if __name__ == "__main__":
    main()