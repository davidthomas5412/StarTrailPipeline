import argparse
import numpy as np
from tqdm import tqdm
from astropy.table import Table
from startrail.paths import registration_dir, valid_table, adjust_table
from numpy.fft import fft2, ifft2
from copy import deepcopy
from startrail.api import Survey
from astropy.wcs import WCS
from scipy.signal import correlate


surv = Survey.get_core_survey()
t = Table.read(adjust_table)

PIX2DEG = 7.285e-5
NUMCCDS = 61

def guess(seq_ind, exp_ind):
    ind = np.where((t['seq'] == seq_ind) * (t['exp'] == exp_ind) * (t['ccd'] == 30))[0][0]
    return t[ind]['ra']

def conv(a,b):
    return np.real(ifft2(fft2(b) * fft2(a)))

def fast_correct(seqInd, expInd, ccdInd):
    seq = surv.sequences[seqInd]
    exp = seq.exposures[expInd]
    ccd = exp.ccds[ccdInd]
    reg = np.loadtxt(f'{registration_dir}/merged_{int(seq.seconds)}.csv', skiprows=1, delimiter=',')

    guessRA = guess(seqInd,expInd)
    guessDEC = 0

    h = deepcopy(seq.exposures[0].ccds[ccdInd].header)
    sky = h['AVSKY'] if 'AVSKY' in h else np.median(ccd.image)
    for key in ['CRVAL1', 'CENRA1', 'COR1RA1', 'COR2RA1', 'COR3RA1', 'COR4RA1']:
        h[key] += guessRA
    
    for key in ['CRVAL2', 'CENDEC1', 'COR1DEC1', 'COR2DEC1', 'COR3DEC1', 'COR4DEC1']:
        h[key] += guessDEC

    raBufferPix = 0
    decBufferPix = 0
    raBuffer = raBufferPix * PIX2DEG
    decBuffer = decBufferPix * PIX2DEG
        
    maxRA = max([h[x] for x in ['COR{}RA1'.format(y) for y in range(1,5)]])
    minRA = min([h[x] for x in ['COR{}RA1'.format(y) for y in range(1,5)]])
    maxDec = max([h[x] for x in ['COR{}DEC1'.format(y) for y in range(1,5)]])
    minDec = min([h[x] for x in ['COR{}DEC1'.format(y) for y in range(1,5)]])

    mask = (reg[:,1] > minRA - raBuffer) * (reg[:,1] < maxRA + raBuffer) *\
        (reg[:,2] > minDec - decBuffer) * (reg[:,2] < maxDec + decBuffer)
    clip = reg[mask]
    clip = clip[np.argsort(clip[:,3])][-100:] # 100 brightest

    wcs = WCS(h)
    pix = wcs.all_world2pix(np.array([clip[:,1], clip[:,2]]).T, 1)

    orig = ccd.image - sky
    n,m = orig.shape
    ext = np.zeros((n+2*raBufferPix, m+2*decBufferPix))
    base = np.zeros((n+2*raBufferPix, m+2*decBufferPix))
    base2 = np.zeros((n+2*raBufferPix, m+2*decBufferPix))
    kernel = np.zeros((n+2*raBufferPix, m+2*decBufferPix))
    xcent = kernel.shape[0] // 2
    ycent = kernel.shape[1] // 2    
    
    trail_length = int(np.cos(seq.dec * np.pi / 180) * exp.header['EXPTIME'] * 15 / 0.263) + 1
    trail_width = 20
    
    # what is the correct space
    kernel[xcent:xcent+trail_length, ycent-(trail_width // 2):ycent+(trail_width // 2)] = 1
    
    for y,x in pix.astype('int'):
        if 0 <= y+decBufferPix < m+2*decBufferPix and 0 <= x+raBufferPix < n+2*raBufferPix :
            base[x+raBufferPix, y+decBufferPix] = 1
    
    res = conv(base, kernel)
    res = np.roll(np.roll(res, xcent, axis=0), ycent, axis=1)

    normed = np.minimum(np.maximum(0, orig), 100)
    corr = correlate(normed, res, mode='same')
    mm = np.unravel_index(corr.argmax(), corr.shape)
    
    deltax = mm[1] - corr.shape[1] // 2
    deltay = mm[0] - corr.shape[0] // 2
    
    deltaRA = guessRA - deltay * PIX2DEG
    deltaDEC = guessDEC + deltax * PIX2DEG

    return deltaRA, deltaDEC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-beg", help="beginning seq_ind", type=int, required=True)
    parser.add_argument("-end", help="ending seq_ind", type=int, required=True)
    args = parser.parse_args()

    for seq_ind in tqdm(range(args.beg, args.end)):
        for exp_ind in tqdm(range(1, len(surv.sequences[seq_ind]))):
            table = Table(names=['seq', 'exp', 'ccd', 'ra', 'dec'], dtype=('i4', 'i4', 'i4', 'f4', 'f4'))
            for ccd_ind in range(NUMCCDS):
                d_ra, d_dec = fast_correct(seq_ind, exp_ind, ccd_ind)
                table.add_row([seq_ind, exp_ind, ccd_ind, d_ra, d_dec])
            table.write(f'{registration_dir}/adjust_core_{seq_ind}_{exp_ind}.csv', overwrite=True)