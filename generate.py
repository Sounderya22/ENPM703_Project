import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

def create_exact_100_humanoids_structure(output_filename="100_humanoids_exact.xml"):
    """
    Create an exact replica of the 100 Humanoids XML structure
    """
    
    # Create the root element with proper attributes
    root = ET.Element("mujoco")
    root.set("model", "100 Humanoids")
    
    # Add license comment
    license_comment = """ Copyright 2021 DeepMind Technologies Limited

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

         http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
"""
    
    # Add option
    option = ET.SubElement(root, "option")
    option.set("timestep", "0.005")
    
    # Add size
    size = ET.SubElement(root, "size")
    size.set("memory", "100M")
    
    # Create asset section
    asset = ET.SubElement(root, "asset")
    
    # Add skybox texture
    texture_skybox = ET.SubElement(asset, "texture")
    texture_skybox.set("type", "skybox")
    texture_skybox.set("builtin", "gradient")
    texture_skybox.set("rgb1", ".3 .5 .7")
    texture_skybox.set("rgb2", "0 0 0")
    texture_skybox.set("width", "512")
    texture_skybox.set("height", "512")
    
    # Add floor texture
    texture_floor = ET.SubElement(asset, "texture")
    texture_floor.set("name", "floor")
    texture_floor.set("type", "2d")
    texture_floor.set("builtin", "checker")
    texture_floor.set("width", "512")
    texture_floor.set("height", "512")
    texture_floor.set("rgb1", ".1 .2 .3")
    texture_floor.set("rgb2", ".2 .3 .4")
    
    # Add floor material
    material_floor = ET.SubElement(asset, "material")
    material_floor.set("name", "floor")
    material_floor.set("texture", "floor")
    material_floor.set("texrepeat", "1 1")
    material_floor.set("texuniform", "true")
    material_floor.set("reflectance", ".2")
    
    # Add model references
    model_humanoid = ET.SubElement(asset, "model")
    model_humanoid.set("name", "humanoid")
    model_humanoid.set("file", "humanoid.xml")
    
    model_double = ET.SubElement(asset, "model")
    model_double.set("name", "double")
    model_double.set("file", "2_humanoids_minimal.xml")
    
    # Create visual section
    visual = ET.SubElement(root, "visual")
    
    # Add visual map
    visual_map = ET.SubElement(visual, "map")
    visual_map.set("force", "0.1")
    visual_map.set("zfar", "30")
    
    # Add visual rgba
    visual_rgba = ET.SubElement(visual, "rgba")
    visual_rgba.set("haze", "0.15 0.25 0.35 1")
    
    # Add visual quality
    visual_quality = ET.SubElement(visual, "quality")
    visual_quality.set("numslices", "16")
    visual_quality.set("numstacks", "8")
    
    # Add visual global
    visual_global = ET.SubElement(visual, "global")
    visual_global.set("offwidth", "800")
    visual_global.set("offheight", "800")
    
    # Create worldbody section
    worldbody = ET.SubElement(root, "worldbody")
    
    # Add floor geom
    geom_floor = ET.SubElement(worldbody, "geom")
    geom_floor.set("name", "floor")
    geom_floor.set("size", "10 10 .05")
    geom_floor.set("type", "plane")
    geom_floor.set("material", "floor")
    geom_floor.set("condim", "3")
    
    # Add directional light
    light_directional = ET.SubElement(worldbody, "light")
    light_directional.set("directional", "true")
    light_directional.set("diffuse", ".9 .9 .9")
    light_directional.set("specular", "0.1 0.1 0.1")
    light_directional.set("pos", "0 0 5")
    light_directional.set("dir", "0 0 -1")
    light_directional.set("castshadow", "true")
    
    # Add spotlight
    light_spotlight = ET.SubElement(worldbody, "light")
    light_spotlight.set("name", "spotlight")
    light_spotlight.set("mode", "targetbodycom")
    light_spotlight.set("target", "world")
    light_spotlight.set("diffuse", "1 1 1")
    light_spotlight.set("specular", "0.3 0.3 0.3")
    light_spotlight.set("pos", "-6 -6 4")
    light_spotlight.set("cutoff", "60")
    
    # Add replication structure
    replicate_outer = ET.SubElement(worldbody, "replicate")
    replicate_outer.set("count", "1")
    replicate_outer.set("euler", "0 0 36")
    replicate_outer.set("sep", "-")
    
    # Add frame
    frame = ET.SubElement(replicate_outer, "frame")
    frame.set("pos", "1.2 0 0")
    
    # Add inner replication
    replicate_inner = ET.SubElement(frame, "replicate")
    replicate_inner.set("count", "4")
    replicate_inner.set("euler", "0 0 17")
    replicate_inner.set("sep", "-")
    replicate_inner.set("offset", "0.6 0 0")
    
    # Add attach elements
    attach_humanoid = ET.SubElement(replicate_inner, "attach")
    attach_humanoid.set("model", "humanoid")
    attach_humanoid.set("body", "torso")
    attach_humanoid.set("prefix", "_")
    
    attach_double = ET.SubElement(replicate_inner, "attach")
    attach_double.set("model", "double")
    attach_double.set("prefix", "double_")
    
    # Convert to string
    rough_string = ET.tostring(root, encoding='utf-8')
    
    # Add the license comment at the beginning
    xml_content = f'<!--{license_comment}-->\n'.encode('utf-8') + rough_string
    
    # Parse and pretty print
    reparsed = minidom.parseString(xml_content)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Remove the extra XML declaration that minidom adds
    lines = pretty_xml.split('\n')
    # Keep only one XML declaration
    xml_declaration = '<?xml version="1.0" ?>'
    content_lines = [line for line in lines if not line.startswith('<?xml')]
    final_xml = xml_declaration + '\n' + '\n'.join(content_lines)
    
    # Write to file
    output_path = os.path.join("new_models", output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_xml)
    
    print(f"Exact 100 Humanoids structure saved as: {output_path}")
    return output_path

def verify_model_files():
    """
    Verify that the required model files exist and have the correct structure
    """
    required_files = [
        "new_models/humanoid.xml",
        "new_models/2_humanoids_minimal.xml"
    ]
    
    print("Verifying model files:")
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
            # Check if the file has a body named "torso"
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                worldbody = root.find('worldbody')
                if worldbody is not None:
                    # Look for a body with name "torso"
                    torso_body = None
                    for body in worldbody.findall('.//body'):
                        if body.get('name') == 'torso':
                            torso_body = body
                            break
                    
                    if torso_body:
                        print(f"  ✓ Contains body 'torso'")
                    else:
                        print(f"  ⚠ No body named 'torso' found")
                        print(f"  Available bodies:")
                        for body in worldbody.findall('.//body'):
                            body_name = body.get('name')
                            if body_name:
                                print(f"    - {body_name}")
            except Exception as e:
                print(f"  ⚠ Could not parse {file_path}: {e}")
        else:
            print(f"✗ {file_path} missing")
    
    return all(os.path.exists(f) for f in required_files)

    

if __name__ == "__main__":
    print("=== Creating Exact 100 Humanoids Structure ===")
    
    # # First, verify or create the required model files
    # if not verify_model_files():
    #     print("\nCreating required model files...")
    #     create_fallback_humanoid_with_torso()
    #     create_fallback_double_humanoids()
    #     verify_model_files()
    
    # Create the exact structure
    output_file = create_exact_100_humanoids_structure()
    
    print(f"\n=== Testing the generated file ===")
    print(f"Run: python simple_test.py {output_file}")
    
    print(f"\n=== If there are attachment errors ===")
    print("The 'attach' elements require bodies named 'torso' in the model files.")
    print("If you get errors, check that humanoid.xml has a body named 'torso'")