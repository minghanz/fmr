"""Check whether the given csv perturbation file satisfies some properties"""
import csv
import numpy as np

path = "pert_010.csv"

with open(path) as f:
    reader = csv.reader(f)
    vmax = 0
    vmax_axis = 0
    for line in reader:
        line = [float(x) for x in line]
        r = np.array(line[:3])
        v = np.array(line[3:])
        # print(np.linalg.norm(r)/np.pi*180)    # 10
        vnorm = np.linalg.norm(v)
        v_axis = np.amax(v)
        # print(vnorm)
        if vnorm > vmax:
            vmax=vnorm
        if v_axis > vmax_axis:
            vmax_axis=v_axis
        # print(line)
    print(vmax)
    print(vmax_axis)