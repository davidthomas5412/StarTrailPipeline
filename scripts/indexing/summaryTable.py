import os
from tqdm import tqdm
from startrail.paths import data_dir, summary_table
from astropy.table import Table
from astropy.io import fits

keys = ['TELSTAT', 'EXPTIME', 'CENTRA', 'CENTDEC', 'TIME-OBS', 'DIMM2SEE', 'DATE-OBS', 'SKYSUB', 'AIRMASS']
t = Table(names=['fname']+keys+['BAND'], dtype=['object','S','f4','f4','f4','S','S','S','f4','f4','S'])
for fname in tqdm(os.listdir(data_dir)):
    if 'registration' in fname:
        continue
    head = fits.open(os.path.join(data_dir, fname))[0].header
    if head['PRODTYPE'] == 'image' and head['PROCTYPE'] == 'SkySub':
        band = head['BAND']
        vals = [fname] + list(map(lambda x: head[x], keys)) + [band.strip()]
        t.add_row(vals)
t.write(summary_table, format='csv', overwrite=True)