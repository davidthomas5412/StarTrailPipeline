from os.path import join

science_dir = '/gpfs/slac/kipac/fs1/u/dthomas5/science/'
data_dir = join(science_dir, 'data')
registration_dir = join(data_dir, 'registration')
summary_table = join(science_dir, 'summary.csv')
valid_table = join(science_dir, 'valid2.csv')
adjust_table = join(registration_dir, 'adjustments6.csv')