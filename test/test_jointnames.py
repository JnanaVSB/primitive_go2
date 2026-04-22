import mujoco
model = mujoco.MjModel.from_xml_path("go2/scene.xml")

# List all body names
for i in range(model.nbody):
    print(i, model.body(i).name)

# List all site names
for i in range(model.nsite):
    print(i, model.site(i).name)