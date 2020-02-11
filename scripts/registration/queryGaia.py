import os
from astroquery.gaia import Gaia
from startrail.api import Survey
from startrail.paths import registration_dir


surv = Survey.get_core_survey()
for seq_ind in [69,73,74,75,76,77,78]:
    seq = surv.sequences[seq_ind]
    ra = seq.ra
    dec = seq.dec
    rad = 1.4
    
    if seq.band == 'r':
        phots = 'phot_bp_mean_flux,phot_bp_mean_flux_error'
        cond = 'phot_bp_mean_flux > 3000'
    else:
        phots = 'phot_rp_mean_flux,phot_rp_mean_flux_error'
        cond = 'phot_rp_mean_flux > 3000'
    
    q1 = f"""SELECT source_id,ra,dec,{phots}
           FROM gaiadr2.gaia_source 
           WHERE 1=CONTAINS(POINT('ICRS',ra,dec), CIRCLE('ICRS',{ra},{dec},{rad})) 
           AND {cond}"""
    fname1 = os.path.join(registration_dir, f'seq_{seq_ind}_3000_1.csv')

    ra = ra + 0.7
    q2 = f"""SELECT source_id,ra,dec,{phots}
           FROM gaiadr2.gaia_source 
           WHERE 1=CONTAINS(POINT('ICRS',ra,dec), CIRCLE('ICRS',{ra},{dec},{rad})) 
           AND {cond}"""
    fname2 = os.path.join(registration_dir, f'seq_{seq_ind}_3000_2.csv')

    # skip already successful
    if not os.path.exists(fname1):
        print(f'Trying: {fname1}')
        try:
            job = Gaia.launch_job_async(query=q1,
                                output_file=fname1,
                                output_format='csv',
                                verbose=True,
                                dump_to_file=True,
                                background=False)
        except Exception as e:
            print('-----------------')
            print('Exception!')
            print(f'query: {q1}')
            print(f'exception: {e}')
            print('-----------------')

    
    if not os.path.exists(fname2):
        print(f'Trying: {fname2}')
        try:
            job = Gaia.launch_job_async(query=q2,
                                output_file=fname2,
                                output_format='csv',
                                verbose=True,
                                dump_to_file=True,
                                background=False)
        except Exception as e:
            print('-----------------')
            print('Exception!')
            print(f'query: {q2}')
            print(f'exception: {e}')
            print('-----------------')
    print(f'Finished seq: {seq_ind}')

print('Done!')



