import os
import numpy as np
from startrail.paths import registration_dir
from astropy.table import Table, vstack
from startrail.api import Survey
from astropy.io import fits
from tqdm import tqdm

def get(seq, exp, ccd):
    path = f'{registration_dir}/adjust_core_{seq}_{exp}.csv'
    try:
        tab = Table.read(path)
        ind = np.where((tab['exp'] == exp) * (tab['ccd'] == ccd))[0][0]
        return tab[ind]['ra'], tab[ind]['dec']
    except Exception: 
        raise RuntimeError(f'{seq}, {exp}, {ccd}')

if __name__ == '__main__':
    ### Include all the relevant header fields
    exp_corners = ['CENTRA', 'CORN1RA', 'CORN2RA', 'CORN3RA', 'CORN4RA','CENTDEC', 'CORN1DEC', 'CORN2DEC', 'CORN3DEC', 'CORN4DEC']
    ccd_corners = ['CENRA1', 'COR1RA1', 'COR2RA1', 'COR3RA1', 'COR4RA1', 'CENDEC1', 'COR1DEC1', 'COR2DEC1', 'COR3DEC1', 'COR4DEC1']
    wcs_keys = ['PV1_7','PV2_8','PV2_9','CD1_1','PV2_0','PV2_1','PV2_2','PV2_3','PV2_4','PV2_5','PV2_6','PV2_7','PV1_6','PV2_10','PV1_4','PV1_3','PV1_2','PV1_1','PV1_0','PV1_9','PV1_8','CD1_2','PV1_5','CD2_1','CD2_2','PV1_10','CRVAL1','CRVAL2']
    core_keys = ['seq', 'exp', 'ccd', 'ra', 'dec']
    names = core_keys + exp_corners + ccd_corners + wcs_keys
    dtype = ['i4', 'i4', 'i4'] + ['f8'] * (len(names) - 3) # note the f8 is important
    nt = Table(names=names, dtype=dtype)

    surv = Survey.get_core_survey()

    prevSeq = None
    for seq in tqdm(range(79)):
        f = fits.open(surv.sequences[seq].exposures[0].fname)
        eh = f[0].header
        for exp in range(1, len(surv.sequences[seq])):
            for ccd in range(61):
                ra, dec = get(seq, exp, ccd)
                h = f[ccd + 1].header
                nrow = [seq, exp, ccd, ra, dec]
                nrow += [eh[k] for k in exp_corners]
                nrow += [h[k] for k in ccd_corners]
                nrow += [h[k] for k in wcs_keys]
                nt.add_row(nrow)

    nt.sort(['seq','exp','ccd'])
    nt.write(f'{registration_dir}/adjustments6.csv', overwrite=True)