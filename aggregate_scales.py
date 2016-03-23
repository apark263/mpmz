from neon.util.persist import save_obj, load_obj
import numpy as np
import sys
from glob import glob
import os

prefix_dir = sys.argv[1]

probfiles = glob(os.path.join(prefix_dir, 'm*', '*.pkl'))
print probfiles

probs = np.zeros((len(probfiles), 10000, 100), dtype=np.float32)
for i, f in enumerate(probfiles):
    probs[i, :, :] = load_obj(f)
aggProbs = probs.mean(axis=0)

vlabels = np.loadtxt('labels.txt').astype(np.int32)[:, np.newaxis]
top1err = 1 - len(np.where(np.argmax(aggProbs, axis=1) == vlabels[:, 0])[0]) / 10000.
top5err = 1 - len(np.where(np.argpartition(-aggProbs, 5)[:, :5] == vlabels)[0]) / 10000.
# import pdb; pdb.set_trace()
print('Top 1 error {}'.format(top1err))
print('Top 5 error {}'.format(top5err))
