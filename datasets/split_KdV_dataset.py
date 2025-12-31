import sys
import numpy as np

if __name__ == "__main__":
    kdv_path = sys.argv[1]
    save_d1_path = sys.argv[2]
    save_d1_path_simplified = sys.argv[3]
    save_d2_path = sys.argv[4]
    data = np.load(kdv_path)
    d1_data = {
        "features": data['solutions'][:, :1],
        "targets": data['solutions'][:, -1:],
        "coordinates": np.expand_dims(data['x'], 0)
    }
    np.savez(save_d1_path, **d1_data)
    del d1_data

    d1_data = {
        "features": data['solutions'][:, :1],
        "targets": data['solutions'][:, ::2][:, 127:128],
        "coordinates": np.expand_dims(data['x'], 0)
    }
    np.savez(save_d1_path_simplified, **d1_data)
    del d1_data


    d2_data = {
        "features": np.expand_dims(np.stack([data['solutions'][:, 0, ::2],]*(data['solutions'].shape[-1]//2), axis=1), 1),
        "targets": np.expand_dims(data['solutions'][:, ::2, ::2], 1),
        "coordinates": np.meshgrid(data['t'][::2], data['x'][::2], indexing='ij')
    }
    np.savez(save_d2_path, **d2_data)