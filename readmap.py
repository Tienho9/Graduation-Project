import numpy as np

data = np.load("saved_map.npy", allow_pickle=True).item()

grid = data["grid"]
start = data["start"]
goal = data["goal"]
targets = data["targets"]
