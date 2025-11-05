# diagnose_and_fix.py
import xml.etree.ElementTree as ET
import os

def diagnose_model_structure():
    """
    Diagnose what bodies are actually available in the model files
    """
    print("=== DIAGNOSING MODEL FILES ===")
    
    model_files = [
        "new_models/humanoid.xml", 
        "new_models/2_humanoids_minimal.xml"
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"\n📁 Analyzing: {file_path}")
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                worldbody = root.find('worldbody')
                
                if worldbody is not None:
                    bodies = worldbody.findall('.//body')
                    print(f"  Found {len(bodies)} body elements:")
                    
                    for i, body in enumerate(bodies):
                        body_name = body.get('name', f'unnamed_body_{i}')
                        body_pos = body.get('pos', 'no position')
                        print(f"    🔹 Body: '{body_name}' at position {body_pos}")
                        
                        # Check for geoms in this body
                        geoms = body.findall('geom')
                        if geoms:
                            for geom in geoms:
                                geom_name = geom.get('name', 'unnamed_geom')
                                geom_type = geom.get('type', 'unknown')
                                print(f"        📦 Geom: '{geom_name}' ({geom_type})")
                        
                        # Check for joints in this body
                        joints = body.findall('joint')
                        if joints:
                            for joint in joints:
                                joint_name = joint.get('name', 'unnamed_joint')
                                joint_type = joint.get('type', 'unknown')
                                print(f"        🔗 Joint: '{joint_name}' ({joint_type})")
                
            except Exception as e:
                print(f"  ❌ Error parsing {file_path}: {e}")
        else:
            print(f"  ❌ File not found: {file_path}")

def find_correct_body_names():
    """
    Find what body names we should use for attachment
    """
    print("\n=== FINDING CORRECT ATTACHMENT BODIES ===")
    
    recommendations = {}
    
    for file_path in ["new_models/humanoid.xml", "new_models/2_humanoids_minimal.xml"]:
        if os.path.exists(file_path):
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                worldbody = root.find('worldbody')
                
                if worldbody is not None:
                    bodies = worldbody.findall('.//body')
                    if bodies:
                        # Get the root body (usually the one we want to attach to)
                        root_body = bodies[0]
                        root_body_name = root_body.get('name')
                        
                        filename = os.path.basename(file_path)
                        recommendations[filename] = root_body_name
                        print(f"  📋 {filename}: Use body '{root_body_name}' for attachment")
                        
            except Exception as e:
                print(f"  ❌ Could not analyze {file_path}: {e}")
    
    return recommendations

def create_fixed_configuration(recommendations):
    """
    Create the correct configuration based on diagnosis
    """
    print("\n=== CREATING FIXED CONFIGURATION ===")
    
    # Build the corrected FILES_TO_INCLUDE
    corrected_files = []
    
    # For humanoid.xml
    humanoid_body = recommendations.get("humanoid.xml", "torso")
    corrected_files.append({
        "name": "humanoid", 
        "file": "humanoid.xml", 
        "attach_body": humanoid_body, 
        "prefix": "_"
    })
    
    # For 2_humanoids_minimal.xml  
    double_body = recommendations.get("2_humanoids_minimal.xml", "torso")
    corrected_files.append({
        "name": "double", 
        "file": "2_humanoids_minimal.xml", 
        "attach_body": double_body, 
        "prefix": "double_"
    })
    
    print("Corrected FILES_TO_INCLUDE configuration:")
    for model in corrected_files:
        print(f"  {{name: '{model['name']}', file: '{model['file']}', attach_body: '{model['attach_body']}', prefix: '{model['prefix']}'}}")
    
    return corrected_files

def create_working_combined_scene():
    """
    Create a working combined scene with the correct body names
    """
    print("\n=== CREATING WORKING COMBINED SCENE ===")
    
    # First, diagnose to get correct body names
    diagnose_model_structure()
    recommendations = find_correct_body_names()
    corrected_files = create_fixed_configuration(recommendations)
    
    # Now create the corrected XML
    root = ET.Element("mujoco")
    root.set("model", "100 Humanoids")
    
    # Add option
    option = ET.SubElement(root, "option")
    option.set("timestep", "0.005")
    
    # Add size
    size = ET.SubElement(root, "size")
    size.set("memory", "100M")
    
    # Create asset section
    asset = ET.SubElement(root, "asset")
    
    # Add textures and materials
    texture_skybox = ET.SubElement(asset, "texture")
    texture_skybox.set("type", "skybox")
    texture_skybox.set("builtin", "gradient")
    texture_skybox.set("rgb1", ".3 .5 .7")
    texture_skybox.set("rgb2", "0 0 0")
    texture_skybox.set("width", "512")
    texture_skybox.set("height", "512")
    
    texture_floor = ET.SubElement(asset, "texture")
    texture_floor.set("name", "floor")
    texture_floor.set("type", "2d")
    texture_floor.set("builtin", "checker")
    texture_floor.set("width", "512")
    texture_floor.set("height", "512")
    texture_floor.set("rgb1", ".1 .2 .3")
    texture_floor.set("rgb2", ".2 .3 .4")
    
    material_floor = ET.SubElement(asset, "material")
    material_floor.set("name", "floor")
    material_floor.set("texture", "floor")
    material_floor.set("texrepeat", "1 1")
    material_floor.set("texuniform", "true")
    material_floor.set("reflectance", ".2")
    
    # Add model references
    for model_config in corrected_files:
        model_elem = ET.SubElement(asset, "model")
        model_elem.set("name", model_config["name"])
        model_elem.set("file", model_config["file"])
    
    # Create worldbody section
    worldbody = ET.SubElement(root, "worldbody")
    
    # Add floor
    geom_floor = ET.SubElement(worldbody, "geom")
    geom_floor.set("name", "floor")
    geom_floor.set("size", "10 10 .05")
    geom_floor.set("type", "plane")
    geom_floor.set("material", "floor")
    geom_floor.set("condim", "3")
    
    # Add lights
    light_directional = ET.SubElement(worldbody, "light")
    light_directional.set("directional", "true")
    light_directional.set("diffuse", ".9 .9 .9")
    light_directional.set("specular", "0.1 0.1 0.1")
    light_directional.set("pos", "0 0 5")
    light_directional.set("dir", "0 0 -1")
    light_directional.set("castshadow", "true")
    
    # Replication structure
    replicate_outer = ET.SubElement(worldbody, "replicate")
    replicate_outer.set("count", "1")
    replicate_outer.set("euler", "0 0 36")
    replicate_outer.set("sep", "-")
    
    frame = ET.SubElement(replicate_outer, "frame")
    frame.set("pos", "1.2 0 0")
    
    replicate_inner = ET.SubElement(frame, "replicate")
    replicate_inner.set("count", "2")
    replicate_inner.set("euler", "0 0 17")
    replicate_inner.set("sep", "-")
    replicate_inner.set("offset", "0.6 0 0")
    
    # Add attach elements with CORRECT body names
    for model_config in corrected_files:
        attach_elem = ET.SubElement(replicate_inner, "attach")
        attach_elem.set("model", model_config["name"])
        attach_elem.set("body", model_config["attach_body"])
        attach_elem.set("prefix", model_config["prefix"])
        print(f"  ✅ Attaching {model_config['name']} to body '{model_config['attach_body']}'")
    
    # Save the file
    from xml.dom import minidom
    rough_string = ET.tostring(root, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    output_path = "new_models/fixed_100_humanoids.xml"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    print(f"\n✅ Fixed combined scene saved as: {output_path}")
    return output_path

def quick_fix_existing_file():
    """
    Quick fix: Modify the existing file with correct body names
    """
    print("\n=== QUICK FIX FOR EXISTING FILE ===")
    
    input_file = "new_models/100_humanoids_combined.xml"
    output_file = "new_models/quick_fixed_humanoids.xml"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
        
        # Find all attach elements
        for attach in root.findall('.//attach'):
            model_name = attach.get('model')
            current_body = attach.get('body')
            
            # Fix the body for double model
            if model_name == "double" and current_body == "torso":
                # Try common alternative body names
                alternative_bodies = ["root", "torso1", "pelvis", "base"]
                attach.set('body', 'root')  # Use 'root' as a common default
                print(f"  🔧 Changed attach body for 'double' from 'torso' to 'root'")
        
        # Save the fixed file
        tree.write(output_file)
        print(f"✅ Quick-fixed file saved as: {output_file}")
        
    except Exception as e:
        print(f"❌ Error fixing file: {e}")

if __name__ == "__main__":
    print("🛠️  MuJoCo Attachment Body Fixer")
    print("=" * 50)
    
    # Option 1: Create a completely new corrected file
    new_file = create_working_combined_scene()
    
    # Option 2: Quick fix the existing file
    quick_fix_existing_file()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Test the new file: python viewer.py {new_file}")
    print(f"2. If it works, update your FILES_TO_INCLUDE configuration with the correct body names")
    print(f"3. If not, run the diagnosis again and check the actual body names in your model files")
