import os
import numpy as np
from startrail.paths import registration_dir
from astropy.table import Table, vstack

if __name__ == '__main__':
    t = vstack([Table.read(f'{registration_dir}/{f}') for f in os.listdir(registration_dir) if '_adjusted.csv' in f])

    individual = Table.read(f'{registration_dir}corrections_core_individual.csv')
    individual.sort(['seq', 'exp', 'ccd'])
    for row in individual:
        mask = (t['seq'] == row['seq']) * (t['exp'] == row['exp']) * (t['ccd'] == row['ccd'])
        idx = np.where(mask)[0][0]
        t['ra'][idx] = row['ra']
        t['dec'][idx] = row['dec']

    for i in range(4):
        fname = f'{registration_dir}corrections_core_sys_{i}.csv'
        if not os.path.exists(fname):
            print('Does Note Exist: ', fname)
        else:
            sys = Table.read(fname)
            for row in sys:
                mask = (t['seq'] == row['seq']) * (t['exp'] == row['exp']) * (t['ccd'] == row['ccd'])
                idx = np.where(mask)[0][0]
                t['ra'][idx] = row['ra']
                t['dec'][idx] = row['dec']
    t.sort(['seq', 'exp', 'ccd'])
    t.write(f'{registration_dir}adjustments.csv', overwrite=True)