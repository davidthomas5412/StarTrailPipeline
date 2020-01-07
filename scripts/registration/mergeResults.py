import os
import numpy as np
from startrail.api import Survey
from startrail.paths import registration_dir

surv = Survey.getCoreSurvey()

for i,seq in enumerate(surv.sequences):
	pair = [x for x in os.listdir(registration_dir) if str(seq.seconds) in x]
	if len(pair) != 2:
		raise RuntimeError('Cannot find pair!')
	
	f1 = os.path.join(registration_dir, pair[0])
	f2 = os.path.join(registration_dir, pair[1])
	
	with open(f1, 'r') as r:
		header = r.readline()
	data1 = np.loadtxt(f1, skiprows=1, delimiter=',')
	data2 = np.loadtxt(f2, skiprows=1, delimiter=',')
	# mask1 = [seq.contains(x,y) for x,y in zip(data1[:,1], data1[:,2])]
	# mask2 = [seq.contains(x,y) for x,y in zip(data2[:,1], data2[:,2])]

	# merged = np.unique(np.vstack([data1[mask1], data2[mask2]]), axis=0)
	merged = np.unique(np.vstack([data1, data2]), axis=0)
	f = os.path.join(registration_dir, f'merged_{int(seq.seconds)}.csv')
	np.savetxt(f, merged, header=header, delimiter=',')
	print(f'{i}/{len(surv.sequences)}')
