# simple_test.py
import mujoco
import mujoco.viewer
import time
import os
import sys

def test_mujoco_file(xml_file):
    """Test if a MuJoCo XML file can be loaded"""
    print(f"Testing: {xml_file}")
    
    if not os.path.exists(xml_file):
        print(f"File does not exist: {xml_file}")
        return False
    
    try:
        # Now try to load in MuJoCo
        model = mujoco.MjModel.from_xml_path(xml_file)
        data = mujoco.MjData(model)
        
        print(f"✓ MuJoCo loads successfully!")
        print(f"  Bodies: {model.nbody}, Joints: {model.njnt}, Geoms: {model.ngeom}")
        
        # Try to run simulation briefly
        print("Running simulation for 3 seconds...")
        print("Press ESC to exit early")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            while viewer.is_running() and (time.time() - start_time) < 3.0:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.001)
        
        print("✓ Simulation completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    else:
        xml_file = "new_models/100_humanoids_exact.xml"
    
    success = test_mujoco_file(xml_file)
    
    if success:
        print("\n🎉 File works correctly!")
    else:
        print("\n❌ File has issues.")