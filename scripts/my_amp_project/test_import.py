import ast

path = "/home/matsuno/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py"

with open(path) as f:
    tree = ast.parse(f.read())

for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "H1_CFG":
                print(node)