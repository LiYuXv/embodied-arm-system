import mujoco

# 1. 读 URDF 并内部转为 MJCF（XML）
model = mujoco.MjModel.from_xml_path("openarm_bimanual.urdf")

# 2. 把 MJCF 保存成独立 XML 文件
with open("openarm_bimanual_mjcf.xml", "w") as f:
    f.write(model.get_xml())