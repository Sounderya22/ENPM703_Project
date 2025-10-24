import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import scipy.linalg
from controllers.lqr_controller import lqr

class Biped:
    def __init__(self, xml_path):
        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.sim_end = 110.0

        self.arms = None  # "fixed"

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

        # Get the LQR Controller Value K
        K, dq = lqr(self.model, self.data, qpos0, ctrl0)

        # Perturbations 
        CTRL_STD, perturb = self.noise()

        return qpos0, ctrl0, K, dq, CTRL_STD, perturb

    def controller(self, dq, qpos0, ctrl0, K, CTRL_STD, perturb, step):
        """
        LQR controller
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

    def simulate(self):
        step = 0
        qpos0, ctrl0, K, dq, CTRL_STD, perturb = self.reset()
        
        while not glfw.window_should_close(self.window):
            simstart = self.data.time

            # Apply periodic external forces
            x = 10 * int(self.data.time / 10)
            y = x + 0.2
            if self.data.time >= x and self.data.time <= y:
                self.apply_external_forces([55.0 + 2 * x / 10, 0.0, 0.0])
            else:
                self.apply_external_forces([0.0, 0.0, 0.0])
            
            while (self.data.time - simstart < 1.0 / 60.0):
                # Step simulation environment
                mj.mj_step(self.model, self.data)

                # Apply control
                self.controller(dq, qpos0, ctrl0, K, CTRL_STD, perturb, step)
                step += 1
                if step >= 2390:
                    step = 0

            if self.data.time >= self.sim_end:
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