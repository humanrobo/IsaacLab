#humanoid_walk.npzの中身を確認するスクリプト
# import numpy as np

# data=np.load(
# "data/humanoid_walk.npz",
# allow_pickle=True
# )

# print(data.files)

# for k in data.files:
#     print(k,data[k].shape)
import numpy as np

data=np.load(
    "data/humanoid_walk.npz",
    allow_pickle=True
)

print(data["dof_names"])
print(data["body_names"])