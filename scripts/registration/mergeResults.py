import os
import numpy as np
from startrail.api import Survey
from startrail.paths import registration_dir
from astropy.table import Table, join


surv = Survey.get_core_survey()

for i in [69,73,74,75,76,77,78]:
	seq = surv.sequences[i]
	pair_500 = [x for x in os.listdir(registration_dir) if (str(seq.index) in x) and ('_500_' in x) and ('seq' in x)]
	pair_3000 = [x for x in os.listdir(registration_dir) if (str(seq.index) in x) and ('_3000_' in x) and ('seq' in x)]
	
	f1 = os.path.join(registration_dir, pair_500[0])
	f2 = os.path.join(registration_dir, pair_500[1])
	t1 = Table.read(f1)
	t2 = Table.read(f2)
	tj = join(t1, t2)
	tj.write(os.path.join(registration_dir,f'registration_{seq.index}_500.csv'), overwrite=True)

	f1 = os.path.join(registration_dir, pair_3000[0])
	f2 = os.path.join(registration_dir, pair_3000[1])
	t1 = Table.read(f1)
	t2 = Table.read(f2)
	tj = join(t1, t2)
	tj.write(os.path.join(registration_dir,f'registration_{seq.index}_3000.csv'), overwrite=True)
