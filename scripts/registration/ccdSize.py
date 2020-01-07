from startrail.api import Survey
from astropy.table import Table
from startrail.paths import registration_dir
from time import time

surv = Survey.getCoreSurvey()
table = Table(names=['seq', 'exp', 'height', 'width'], dtype=['i4', 'i4', 'i4', 'i4'])
for i in range(len(surv)):
	for j in range(len(surv.sequences[i])):
		height, width = surv.sequences[i].exposures[j].ccds[1].image.shape
		print(time(), i, j, height, width)
		table.add_row([i, j, height, width])

table.write(f'{registration_dir}ccdSize.csv', overwrite=True)