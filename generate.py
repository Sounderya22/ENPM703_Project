import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

# CONFIGURATION VARIABLES - Separate counts per model
CONFIG = {
    # Scene settings
    "model_name": "Custom Humanoids",
    "timestep": "0.005",
    "memory_size": "100M",
    
    # Replication structure
    "outer_euler": "0 0 36", 
    "outer_sep": "-",
    "frame_pos": "1.2 0 0",
    "inner_euler": "0 0 17",
    "inner_sep": "-", 
    "inner_offset": "0.6 0 0",
    
    # Models with SEPARATE COUNTS
    "models": [
        {
            "name": "humanoid",
            "file": "humanoid.xml", 
            "count": 2,                    # ← 3 instances of this model
            "attach_body": "torso",
            "prefix": "_"
        },
        {
            "name": "double",
            "file": "2_humanoids_minimal.xml",
            "count": 3,                    # ← 2 instances of this model
            # NO body attribute - like working file
            "prefix": "double_"  
        }
    ]
}

def create_working_combined_scene():
    """
    Create working_combined.xml with separate counts per model
    """
    
    # Create the root element
    root = ET.Element("mujoco")
    root.set("model", CONFIG["model_name"])
    
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
    option.set("timestep", CONFIG["timestep"])
    
    # Add size
    size = ET.SubElement(root, "size")
    size.set("memory", CONFIG["memory_size"])
    
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
    for model in CONFIG["models"]:
        model_elem = ET.SubElement(asset, "model")
        model_elem.set("name", model["name"])
        model_elem.set("file", model["file"])
    
    # Create visual section
    visual = ET.SubElement(root, "visual")
    
    visual_map = ET.SubElement(visual, "map")
    visual_map.set("force", "0.1")
    visual_map.set("zfar", "30")
    
    visual_rgba = ET.SubElement(visual, "rgba")
    visual_rgba.set("haze", "0.15 0.25 0.35 1")
    
    visual_quality = ET.SubElement(visual, "quality")
    visual_quality.set("numslices", "16")
    visual_quality.set("numstacks", "8")
    
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
    
    # Create SEPARATE REPLICATION for EACH MODEL with its own count
    for model_config in CONFIG["models"]:
        if model_config["count"] > 0:
            # Create replication for this specific model type
            replicate = ET.SubElement(worldbody, "replicate")
            replicate.set("count", str(model_config["count"]))
            replicate.set("euler", CONFIG["inner_euler"])
            replicate.set("sep", CONFIG["inner_sep"])
            replicate.set("offset", CONFIG["inner_offset"])
            
            # Position this model group
            frame = ET.SubElement(replicate, "frame")
            frame.set("pos", CONFIG["frame_pos"])
            frame.set("euler", CONFIG["outer_euler"])
            
            # Attach the model
            attach_elem = ET.SubElement(frame, "attach")
            attach_elem.set("model", model_config["name"])
            attach_elem.set("prefix", model_config["prefix"])
            # Only add body attribute if specified
            if "attach_body" in model_config and model_config["attach_body"]:
                attach_elem.set("body", model_config["attach_body"])
    
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
    output_path = "new_models/working_combined.xml"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_xml)
    
    print(f"✅ working_combined.xml saved successfully!")
    
    # Print summary
    total_instances = sum(model["count"] for model in CONFIG["models"])
    print(f"📊 Total model instances: {total_instances}")
    for model in CONFIG["models"]:
        print(f"   - {model['name']}: {model['count']} instances")
    
    return output_path

if __name__ == "__main__":
    create_working_combined_scene()
    print(f"\n🎯 Run: python viewer.py working_combined.xml")