import os
import numpy as np
from astropy.table import Table
from startrail.api import Survey
from startrail.paths import registration_dir

surv = Survey.get_core_survey()
for seq_ind in range(len(surv)):
	print(seq_ind)
	key = int(surv.sequences[seq_ind].seconds)
	data500 = np.loadtxt(os.path.join(registration_dir, f'merged_{key}_500.csv') , delimiter=',')
	data3000 = np.loadtxt(os.path.join(registration_dir, f'merged_{key}.csv') , delimiter=',')
	Table(names=['source_id','ra','dec','phot_bp_mean_flux','phot_bp_mean_flux_error'], 
		data=data500).write(os.path.join(registration_dir,f'registration_{seq_ind}_500.csv'), overwrite=True)
	Table(names=['source_id','ra','dec','phot_bp_mean_flux','phot_bp_mean_flux_error'], 
		data=data3000).write(os.path.join(registration_dir,f'registration_{seq_ind}_3000.csv'), overwrite=True)
