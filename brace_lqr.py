#!/usr/bin/python

import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import scipy.linalg

def lqr_brace(model, data, qpos0, ctrl0):    
    nu = model.nu  # Number of actuators (4x the single humanoid)
    R_mat = np.eye(nu)

    nv = model.nv  # Number of DoFs

    # Get the Jacobian for the overall system COM
    mj.mj_resetData(model, data)
    data.qpos = qpos0
    mj.mj_forward(model, data)
    
    # Use the first humanoid's torso as the primary balancing target
    jac_com = np.zeros((3, nv))
    torso1_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'torso1')
    mj.mj_jacSubtreeCom(model, data, jac_com, torso1_body_id)

    # Get Jacobians for all feet to maintain stability
    jac_feet = []
    foot_bodies = ['foot_left1', 'foot_right1', 'foot_left2', 'foot_right2', 
                   'foot_left3', 'foot_right3', 'foot_left4', 'foot_right4']
    
    for foot_body in foot_bodies:
        jac_foot = np.zeros((3, nv))
        foot_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, foot_body)
        if foot_body_id != -1:
            mj.mj_jacBodyCom(model, data, jac_foot, None, foot_body_id)
            jac_feet.append(jac_foot)

    # Balance cost based on COM relative to feet support polygon
    if jac_feet:
        jac_feet_avg = np.mean(jac_feet, axis=0)
       # Replace the Qbalance calculation:
        jac_diff = jac_com - jac_feet_avg

        # Check for NaN and replace with zeros
        if np.any(np.isnan(jac_diff)):
            print("Warning: NaN detected in jac_diff, using zeros")
            jac_diff = np.zeros_like(jac_diff)

        Qbalance = jac_diff.T @ jac_diff
    else:
        Qbalance = np.zeros((nv, nv))

    # Get all joint names
    joint_names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]

    # Get indices into relevant sets of joints
    root_dofs = []
    body_dofs = []
    
    # Find root and body DOFs for all humanoids
    for i in range(nv):
        joint_id = model.dof_jntid[i]
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name and ('root' in joint_name or joint_name in ['root1', 'root2', 'root3', 'root4']):
            root_dofs.append(i)
        else:
            body_dofs.append(i)

    root_dofs = np.array(root_dofs)
    body_dofs = np.array(body_dofs)

    # Identify balancing joints (leg joints for all humanoids)
    balance_dofs = []
    for name in joint_names:
        if name and any(term in name for term in ['hip', 'knee', 'ankle', 'abdomen']):
            dof_adr = model.joint(name).dofadr[0]
            if dof_adr not in root_dofs:
                balance_dofs.append(dof_adr)

    balance_dofs = np.array(balance_dofs)
    other_dofs = np.setdiff1d(body_dofs, balance_dofs)

    # Cost coefficients for brace configuration
    BALANCE_COST = 500    # Lower than single humanoid due to mutual support
    BALANCE_JOINT_COST = 2
    OTHER_JOINT_COST = 0.2
    CONNECTION_COST = 100  # Penalty for connection joint stresses

    # Construct the Qjoint matrix
    Qjoint = np.eye(nv)
    Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joints directly
    Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
    Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

    # Additional cost for maintaining connection stability
    Qconnection = np.zeros((nv, nv))
    # This would ideally penalize relative motion between connected bodies
    
    # Construct the Q matrix for position DoFs
    Qpos = BALANCE_COST * Qbalance + Qjoint + Qconnection

    # No explicit penalty for velocities
    Q = np.block([[Qpos, np.zeros((nv, nv))],
                  [np.zeros((nv, nv)), np.zeros((nv, nv))]])

    # Set the initial state and control
    mj.mj_resetData(model, data)
    data.ctrl = ctrl0
    data.qpos = qpos0

    # Allocate the A and B matrices, compute them
    A = np.zeros((2*nv, 2*nv))
    B = np.zeros((2*nv, nu))
    epsilon = 1e-6
    centered = True
    mj.mjd_transitionFD(model, data, epsilon, centered, A, B, None, None)

    # Solve discrete Riccati equation
    P = scipy.linalg.solve_discrete_are(A, B, Q, R_mat)

    # Compute the feedback gain matrix K
    K = np.linalg.inv(R_mat + B.T @ P @ B) @ B.T @ P @ A

    data.qpos = qpos0

    # Allocate position difference dq
    dq = np.zeros(model.nv)

    return K, dq, P, A, B, Q

def compute_brace_stability_metrics(model, data, P, A, B, K, qpos0, prev_state_error=None, dt=0.0):
    """
    Compute comprehensive stability metrics for the brace LQR-controlled system
    """
    # Current state deviation from desired
    dq_current = np.zeros(model.nv)
    mj.mj_differentiatePos(model, dq_current, 1, qpos0, data.qpos)
    state_error = np.concatenate([dq_current, data.qvel])
    
    # 1. LQR Cost-to-Go
    cost_to_go = state_error.T @ P @ state_error
    
    # 2. Normalized stability score
    stability_score = 1.0 / (1.0 + 0.1 * cost_to_go)
    
    # 3. Eigenvalue stability analysis
    A_cl = A - B @ K
    eigvals = np.linalg.eigvals(A_cl)
    
    # Stability margin
    real_parts = np.real(eigvals)
    stability_margin = np.min(-real_parts) if np.all(real_parts < 0) else 0
    
   # 4. Brace-specific metrics
    # Check connection forces (weld joint constraints)
    if hasattr(data, 'efc_force') and data.efc_force is not None and len(data.efc_force) > 0:
        avg_connection_force = np.linalg.norm(data.efc_force) / len(data.efc_force)
    else:
        avg_connection_force = 0
    
    # 5. Collective COM metrics
    com_positions = []
    com_velocities = []
    for i in range(1, 5):  # Humanoids 1-4
        torso_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f'torso{i}')
        if torso_id != -1:
            com_positions.append(data.subtree_com[torso_id])
            # Extract COM velocity from cvel
            com_velocities.append(np.linalg.norm(data.cvel[torso_id][3:6]))
    
    if com_positions:
        collective_com = np.mean(com_positions, axis=0)
        collective_com_height = collective_com[2]
        avg_com_velocity = np.mean(com_velocities)
    else:
        collective_com_height = 0
        avg_com_velocity = 0
    
    # 6. Control effort
    torque_usage = np.linalg.norm(data.ctrl)
    max_torque = np.max(np.abs(data.ctrl))
    
    # 7. Lyapunov rate of change
    lyapunov_rate = 0.0
    if prev_state_error is not None and dt > 0:
        V_prev = prev_state_error.T @ P @ prev_state_error
        V_current = cost_to_go
        lyapunov_rate = (V_current - V_prev) / dt
    
    # 8. Brace-specific stability score
    connection_stability = 1.0 / (1.0 + 0.1 * avg_connection_force)
    
    comprehensive_score = (
        stability_score * 0.3 +                    # LQR performance
        connection_stability * 0.3 +               # Connection stability
        np.exp(-avg_com_velocity) * 0.2 +          # Collective COM velocity
        np.exp(-torque_usage/20) * 0.2             # Control effort (scaled for 4x humanoids)
    )
    
    return {
        'stability_score': stability_score,
        'comprehensive_score': comprehensive_score,
        'connection_stability': connection_stability,
        'lqr_cost': cost_to_go,
        'stability_margin': stability_margin,
        'collective_com_height': collective_com_height,
        'avg_com_velocity': avg_com_velocity,
        'avg_connection_force': avg_connection_force,
        'torque_usage': torque_usage,
        'max_torque': max_torque,
        'lyapunov_rate': lyapunov_rate,
        'state_error': state_error.copy(),
        'is_stable': stability_margin > 0 and lyapunov_rate <= 0 and connection_stability > 0.8
    }

class Brace:
    def __init__(self, xml_path):
        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.sim_end = 110.0

        # Stability tracking
        self.stability_history = []
        self.prev_state_error = None
        self.stability_data = []

        # Initialize GLFW and visualization
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        self.window = glfw.create_window(1200, 900, "Brace LQR", None, None)
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
        
        # Set camera configuration for better brace view
        self.cam.azimuth = 45
        self.cam.elevation = -20
        self.cam.distance = 12.0
        self.cam.lookat = np.array([0.0, 0.0, 2.0])

        # Install GLFW callbacks
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
        # Find a stable standing configuration for the brace
        qpos0, ctrl0 = self.find_brace_equilibrium()

        # Get the LQR Controller
        K, dq, P, A, B, Q = lqr_brace(self.model, self.data, qpos0, ctrl0)

        # Perturbations 
        CTRL_STD, perturb = self.noise()

        # Reset stability tracking
        self.stability_history = []
        self.prev_state_error = None
        self.stability_data = []

        return qpos0, ctrl0, K, dq, CTRL_STD, perturb, P, A, B

    def find_brace_equilibrium(self):
        """Find equilibrium configuration for the brace system"""
        # Start from default pose
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        
        # Set all humanoids to standing pose
        # This is simplified - in practice you'd need to set each humanoid's pose
        qpos0 = self.data.qpos.copy()
        
        # Use inverse dynamics to find control torques for equilibrium
        self.data.qacc = 0
        mj.mj_inverse(self.model, self.data)
        qfrc0 = self.data.qfrc_inverse.copy()

        # Build actuator moment matrix
        nu = self.model.nu
        nv = self.model.nv
        actuator_moment = np.zeros((nu, nv))

        for i in range(nu):
            mj.mj_resetData(self.model, self.data)
            self.data.qpos = qpos0.copy()
            self.data.ctrl[:] = 0.0
            self.data.ctrl[i] = 1.0
            mj.mj_forward(self.model, self.data)
            actuator_moment[i, :] = self.data.qfrc_actuator.copy()

        # Solve least-squares for control torques
        ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(actuator_moment)
        ctrl0 = ctrl0.flatten()

        # Reset sim with solution
        mj.mj_resetData(self.model, self.data)
        self.data.qpos = qpos0.copy()
        self.data.ctrl = ctrl0.copy()

        return qpos0, ctrl0

    def noise(self):
        nu = self.model.nu

        DURATION = 30
        CTRL_STD = 0.02  # Lower noise for brace system
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

    def controller(self, dq, qpos0, ctrl0, K, CTRL_STD, perturb, step, P, A, B):
        """
        LQR controller for brace system with stability monitoring
        """
        # Get state difference dx
        mj.mj_differentiatePos(self.model, dq, 1, qpos0, self.data.qpos)
        dx = np.hstack((dq, self.data.qvel)).T

        # LQR control law
        self.data.ctrl = ctrl0 - K @ dx

        # Add perturbation
        self.data.ctrl += CTRL_STD * perturb[step]
        
        # Compute stability metrics
        dt = self.model.opt.timestep
        stability_metrics = compute_brace_stability_metrics(
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
              f"Connection: {metrics['connection_stability']:.3f}, "
              f"Comp: {metrics['comprehensive_score']:.3f}, "
              f"LQR Cost: {metrics['lqr_cost']:.1f}, "
              f"COM Vel: {metrics['avg_com_velocity']:.3f}, "
              f"Torque: {metrics['torque_usage']:.2f}, "
              f"Stable: {metrics['is_stable']}")

    def simulate(self):
        step = 0
        qpos0, ctrl0, K, dq, CTRL_STD, perturb, P, A, B = self.reset()
        
        while not glfw.window_should_close(self.window):
            simstart = self.data.time
            
            while (self.data.time - simstart < 1.0 / 60.0):
                # Step simulation environment
                mj.mj_step(self.model, self.data)

                # Apply control with stability monitoring
                stability_metrics = self.controller(dq, qpos0, ctrl0, K, CTRL_STD, perturb, step, P, A, B)
                step += 1
                if step >= len(perturb):
                    step = 0

            if self.data.time >= self.sim_end:
                # Print final stability summary
                self.print_final_stability_summary()
                break

            # Render
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Visualization options
            self.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = 0
            self.opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = 1
            self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
            self.model.vis.map.force = 0.01

            # Update scene and render
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

    def print_final_stability_summary(self):
        """Print comprehensive stability analysis at the end of simulation"""
        if not self.stability_data:
            return
            
        print("\n" + "="*60)
        print("BRACE FINAL STABILITY ANALYSIS")
        print("="*60)
        
        # Calculate statistics
        stability_scores = [d['stability_score'] for d in self.stability_data]
        comp_scores = [d['comprehensive_score'] for d in self.stability_data]
        connection_stabilities = [d['connection_stability'] for d in self.stability_data]
        lqr_costs = [d['lqr_cost'] for d in self.stability_data]
        com_velocities = [d['avg_com_velocity'] for d in self.stability_data]
        torque_usages = [d['torque_usage'] for d in self.stability_data]
        stable_states = [d['is_stable'] for d in self.stability_data]
        
        print(f"Average Stability Score: {np.mean(stability_scores):.3f} ± {np.std(stability_scores):.3f}")
        print(f"Average Connection Stability: {np.mean(connection_stabilities):.3f} ± {np.std(connection_stabilities):.3f}")
        print(f"Average Comprehensive Score: {np.mean(comp_scores):.3f} ± {np.std(comp_scores):.3f}")
        print(f"Average LQR Cost: {np.mean(lqr_costs):.1f} ± {np.std(lqr_costs):.1f}")
        print(f"Average COM Velocity: {np.mean(com_velocities):.3f} ± {np.std(com_velocities):.3f}")
        print(f"Average Torque Usage: {np.mean(torque_usages):.2f} ± {np.std(torque_usages):.2f}")
        print(f"Stable States: {np.sum(stable_states)}/{len(stable_states)} ({np.mean(stable_states)*100:.1f}%)")
        
        # Stability classification
        avg_stability = np.mean(comp_scores)
        if avg_stability > 0.8:
            stability_class = "EXCELLENT"
        elif avg_stability > 0.6:
            stability_class = "GOOD" 
        elif avg_stability > 0.4:
            stability_class = "MARGINAL"
        else:
            stability_class = "POOR"
            
        print(f"Overall Brace Stability: {stability_class}")
        print("="*60)

def main():
    xml_path = "models/brace.xml"  # Update path to your brace XML file
    sim = Brace(xml_path)
    sim.simulate()

if __name__ == "__main__":
    main()