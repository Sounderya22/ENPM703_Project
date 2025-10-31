import mujoco
import mujoco.viewer
from xml.etree import ElementTree as ET

def merge_models(main_xml, secondary_xml, secondary_offset=(1, 0, 0)):
    # Parse both XMLs
    main_tree = ET.parse(main_xml)
    secondary_tree = ET.parse(secondary_xml)

    main_root = main_tree.getroot()
    secondary_root = secondary_tree.getroot()

    # Worldbodies
    main_worldbody = main_root.find("worldbody")
    secondary_worldbody = secondary_root.find("worldbody")

    # Offset all bodies in secondary
    for body in secondary_worldbody.findall("body"):
        body.set("pos", f"{secondary_offset[0]} {secondary_offset[1]} {secondary_offset[2]}")
        main_worldbody.append(body)

    # Merge assets, renaming duplicates
    main_asset = main_root.find("asset")
    secondary_asset = secondary_root.find("asset")

    if secondary_asset is not None:
        if main_asset is None:
            main_asset = ET.SubElement(main_root, "asset")

        existing_names = {a.get("name") for a in main_asset if a.get("name")}
        for asset in secondary_asset:
            name = asset.get("name")
            if name in existing_names:
                # Rename to avoid collision
                new_name = f"{name}_2"
                print(f"Renaming duplicate asset '{name}' -> '{new_name}'")
                asset.set("name", new_name)
            main_asset.append(asset)

    return ET.tostring(main_root, encoding="unicode")

# --- Main ---
robot_xml = "models/humanoid_pyramid.xml"
object_xml = "models/humanoid.xml"

merged_xml = merge_models(robot_xml, object_xml, secondary_offset=(1, 0, 0))
model = mujoco.MjModel.from_xml_string(merged_xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

