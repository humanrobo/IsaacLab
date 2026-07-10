#mujocoがurdfを読み込めるようにするため、urdfのメッシュ情報を消すスクリプト
import xml.etree.ElementTree as ET


src = "data/h1.urdf"
dst = "data/h1_ik_only.urdf"


tree = ET.parse(src)
root = tree.getroot()


# visualとcollisionを削除
for link in root.findall("link"):

    for child in list(link):

        if child.tag in ["visual", "collision"]:
            link.remove(child)


tree.write(dst)

print("saved:", dst)