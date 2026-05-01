import sys
import zarr
z = zarr.open(sys.argv[1], mode="r")
print("action:", z["data"]["actions"].shape, z["data"]["actions"].dtype)
if "expert_mask" in z["data"]:
    print("expert_mask:", z["data"]["expert_mask"].shape, z["data"]["expert_mask"].dtype)
for k in sorted(list(z["data"]["obs"].group_keys()) + list(z["data"]["obs"].array_keys())):
    arr = z["data"]["obs"][k]
    print(f"obs/{k}: {arr.shape} {arr.dtype}")
