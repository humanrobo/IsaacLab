#h1.urdfの中身を確認するスクリプト

import xml.etree.ElementTree as ET

urdf_path = "data/h1.urdf"

tree = ET.parse(urdf_path)
root = tree.getroot()

print("=== Links ===")
for link in root.findall("link"):
    print(link.attrib["name"])

print("\n=== Joints ===")
for joint in root.findall("joint"):
    name = joint.attrib["name"]
    jtype = joint.attrib["type"]

    parent = joint.find("parent").attrib["link"]
    child = joint.find("child").attrib["link"]

    print(f"{name:30s} type={jtype:10s} {parent} -> {child}")