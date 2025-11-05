import mujoco
import mujoco.viewer
import time
import numpy as np
import os

class MuJoCoSimulator:
    def __init__(self, xml_path):
        """
        Initialize MuJoCo simulator with XML file
        
        Args:
            xml_path (str): Path to the XML file
        """
        self.xml_path = xml_path
        self.model = None
        self.data = None
        self.viewer = None
        
    def load_model(self):
        """Load the model from XML file"""
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"XML file not found: {self.xml_path}")
        
        try:
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
            self.data = mujoco.MjData(self.model)
            print(f"✅ Model loaded successfully: {self.xml_path}")
            self.print_model_info()
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def print_model_info(self):
        """Print detailed information about the loaded model"""
        print("\n" + "="*50)
        print("MODEL INFORMATION")
        print("="*50)
        #print(f"Model name: {self.model.name}")
        print(f"Number of bodies: {self.model.nbody}")
        print(f"Number of joints: {self.model.njnt}")
        print(f"Number of actuators: {self.model.nu}")
        #print(f"Number of constraints: {self.model.ne}")
        print(f"Number of geometries: {self.model.ngeom}")
        print(f"Timestep: {self.model.opt.timestep}")
        
        # Print joint names
        if self.model.njnt > 0:
            print(f"\nJoint names:")
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                print(f"  {i}: {joint_name}")
        
        # Print actuator names
        if self.model.nu > 0:
            print(f"\nActuator names:")
            for i in range(self.model.nu):
                actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                print(f"  {i}: {actuator_name}")
    
    def set_initial_state(self, qpos=None, qvel=None):
        """Set initial state of the simulation"""
        if qpos is not None and len(qpos) == self.model.nq:
            self.data.qpos[:] = qpos
        if qvel is not None and len(qvel) == self.model.nv:
            self.data.qvel[:] = qvel
    
    def apply_control(self, control_signal=None):
        """Apply control signals to actuators"""
        if control_signal is None and self.model.nu > 0:
            # Apply zero control by default
            self.data.ctrl[:] = np.zeros(self.model.nu)
        elif control_signal is not None:
            if len(control_signal) == self.model.nu:
                self.data.ctrl[:] = control_signal
            else:
                print(f"Warning: Control signal length {len(control_signal)} doesn't match number of actuators {self.model.nu}")
    
    def run_simulation(self, duration=10.0, control_callback=None):
        """
        Run the simulation
        
        Args:
            duration (float): Simulation duration in seconds
            control_callback (callable): Optional callback function for custom control
        """
        if self.model is None or self.data is None:
            print("❌ Model not loaded. Call load_model() first.")
            return
        
        print(f"\n🚀 Starting simulation for {duration} seconds...")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Configure camera
            viewer.cam.distance = 6.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            
            self.viewer = viewer
            
            start_time = time.time()
            sim_time = 0.0
            
            while viewer.is_running() and sim_time < duration:
                step_start = time.time()
                
                # Apply control
                if control_callback:
                    # Custom control from callback
                    control_signal = control_callback(self.data, sim_time)
                    self.apply_control(control_signal)
                else:
                    # Default random control
                    if self.model.nu > 0:
                        self.data.ctrl[:] = np.random.uniform(-0.1, 0.1, self.model.nu)
                
                # Step the simulation
                mujoco.mj_step(self.model, self.data)
                
                # Update simulation time
                sim_time = self.data.time
                
                # Sync viewer
                viewer.sync()
                
                # Print progress every second
                if int(sim_time) > int(sim_time - self.model.opt.timestep):
                    print(f"Simulation time: {sim_time:.2f}s")
                
                # Rudimentary time keeping
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            
            print("✅ Simulation completed!")
    
    def close(self):
        """Close the viewer if open"""
        if self.viewer:
            self.viewer.close()

# Example usage with custom control
def custom_controller(data, time):
    """Example custom controller that creates oscillating movements"""
    n_actuators = len(data.ctrl)
    control_signal = np.zeros(n_actuators)
    
    # Create sinusoidal control signals
    for i in range(n_actuators):
        frequency = 0.5 + i * 0.1  # Different frequency for each actuator
        amplitude = 0.2
        control_signal[i] = amplitude * np.sin(2 * np.pi * frequency * time)
    
    return control_signal

def main():
    # Path to your XML file
    # xml_path = "models/humanoid_pyramid.xml"
    xml_path = "new_models/working_combined.xml"

    
    # Create simulator instance
    simulator = MuJoCoSimulator(xml_path)
    
    # Load the model
    if simulator.load_model():
        # Run simulation with default random control
        print("\n1. Running with random control...")
        simulator.run_simulation(duration=30.0)
        
        # Run simulation with custom control
        print("\n2. Running with custom oscillating control...")
        simulator.run_simulation(duration=30.0, control_callback=custom_controller)
        
        # Close the simulator
        simulator.close()

if __name__ == "__main__":
    main()