import h5py
import glob
import numpy as np
from glob import glob

BASE_DIR = '/bigdata/shared/L1AnomalyDetection/hChToTauNu_LOWMASS/'
OUT_NAME = '/bigdata/shared/L1AnomalyDetection/hChToTauNu.npy'
list_files = glob(BASE_DIR+"/*.h5")
for i,each in enumerate(list_files):
    if i % 100 == 0:
        print("Processing {}th file".format(i))
    with h5py.File(each,"r") as infile:
        particles = np.asarray(infile['Particles']).astype(np.float16)
        if i == 0:
            all_particles = particles
        else:
            all_particles = np.concatenate((all_particles, particles))

np.save(OUT_NAME, all_particles)
print("Saved to ", OUT_NAME)
