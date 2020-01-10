import argparse
import numpy as np
from tqdm import tqdm
from numpy.fft import fft2, ifft2
from startrail.api import Survey
from startrail.paths import registration_dir
from scipy.signal import correlate
from copy import deepcopy
from astropy.wcs import WCS
from astropy.table import Table

PIX2DEG = 0.263 / 60 / 60
NUMCCDS = 61 # TODO: change back
GUESS_RA_MAP = {
    (59,1): 0.21,
    (59,2): 0.37,
    (59,3): 0.56,
    (63,3): 0.62,
}

def conv(a,b):
    return np.real(ifft2(fft2(b) * fft2(a)))

def fastCorrect(surv, seqInd, expInd, ccdInd):
    seq = surv.sequences[seqInd]
    exp = seq.exposures[expInd]
    ccd = exp.ccds[ccdInd]
    reg = np.loadtxt(f'{registration_dir}merged_{int(seq.seconds)}.csv', skiprows=1, delimiter=',')

    guessRA = GUESS_RA_MAP[(seqInd,expInd)]
    guessDEC = 0

    h = deepcopy(seq.exposures[0].ccds[ccdInd].header)
    sky = h['AVSKY'] if 'AVSKY' in h else np.median(ccd.image)
    for key in ['CRVAL1', 'CENRA1', 'COR1RA1', 'COR2RA1', 'COR3RA1', 'COR4RA1']:
        h[key] += guessRA
    
    for key in ['CRVAL2', 'CENDEC1', 'COR1DEC1', 'COR2DEC1', 'COR3DEC1', 'COR4DEC1']:
        h[key] += guessDEC

    raBufferPix = 900
    decBufferPix = 100
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
    
    trail_length = int(np.cos(seq.centDEC * np.pi / 180) * exp.header['EXPTIME'] * 15 / 0.263) + 1
    trail_width = 20
    kernel[xcent:xcent+trail_length, ycent:ycent+trail_width] = 1
    for y,x in pix.astype('int'):
        if 0 <= y+decBufferPix < m+2*decBufferPix and 0 <= x+raBufferPix < n+2*raBufferPix :
            base[x+raBufferPix, y+decBufferPix] = 1
    
    ext[raBufferPix:-raBufferPix, decBufferPix:-decBufferPix] = orig

    res = conv(base, kernel)
    res = np.roll(np.roll(res, xcent, axis=0), ycent, axis=1)

    normed = np.minimum(np.maximum(0, ext), 10)
    corr = correlate(normed, res, mode='same')
    mm = np.unravel_index(corr.argmax(), corr.shape)
    deltax = mm[1] - corr.shape[1] // 2
    deltay = mm[0] - corr.shape[0] // 2

    deltaRA = guessRA - deltay * PIX2DEG
    deltaDEC = guessDEC - deltax * PIX2DEG

    return deltaRA, deltaDEC

if __name__ == '__main__':
    table = Table(names=['seq', 'exp', 'ccd', 'ra', 'dec'], dtype=('i4', 'i4', 'i4', 'f4', 'f4'))
    surv = Survey.getCoreSurvey()
    for seqInd, expInd in GUESS_RA_MAP:
        for ccdInd in tqdm(range(NUMCCDS)):
            deltaRA, deltaDEC = fastCorrect(surv, seqInd, expInd, ccdInd)
            table.add_row([seqInd, expInd, ccdInd, deltaRA, deltaDEC])    
    table.write(f'{registration_dir}corrections_core_sys_1.csv', overwrite=True)