import os
from startrail.paths import registration_dir
from astropy.table import Table, vstack

if __name__ == '__main__':
    t = vstack([Table.read(f'{registration_dir}/{f}') for f in os.listdir(registration_dir) if 'corrections_core' in f])
    t.write(f'{registration_dir}corrections.csv', overwrite=True)