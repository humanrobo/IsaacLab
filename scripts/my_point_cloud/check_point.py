import torch
import numpy as np

path = "pointcloud.pt"

pc = torch.load(path)

print("type:", type(pc))

if isinstance(pc, dict):
    print("keys:", pc.keys())

    for k, v in pc.items():
        print("\n---", k, "---")
        print("type:", type(v))

        if torch.is_tensor(v):
            print("shape:", v.shape)
            print("dtype:", v.dtype)
            print("min:", v.min())
            print("max:", v.max())

else:
    print("shape:", pc.shape)
    print("dtype:", pc.dtype)

    print("\nfirst 5 points:")
    print(pc[:5])

    pc_np = pc.cpu().numpy()

    print("\nxyz range")
    print("x:", pc_np[:,0].min(), pc_np[:,0].max())
    print("y:", pc_np[:,1].min(), pc_np[:,1].max())
    print("z:", pc_np[:,2].min(), pc_np[:,2].max())

    print("\nNaN:")
    print(np.isnan(pc_np).sum())

    print("Inf:")
    print(np.isinf(pc_np).sum())