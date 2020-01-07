import argparse
import numpy as np
from numpy.fft import fft2, ifft2
from startrail.api import Survey
from startrail.paths import registration_dir
from scipy.signal import correlate
from copy import deepcopy
from astropy.wcs import WCS
from astropy.table import Table

def conv(a,b):
    return np.real(ifft2(fft2(b) * fft2(a)))

def fastCorrect(surv, seqInd, expInd, ccdInd):
    seq = surv.sequences[seqInd]
    exp = seq.exposures[expInd]
    ccd = exp.ccds[ccdInd]
    reg = np.loadtxt(f'{registration_dir}merged_{int(seq.seconds)}.csv', skiprows=1, delimiter=',')
    reg = reg[reg[:,3] > 20000] # flux threshold

    if seqInd == 1:
        expToGuess = {
            1: 0.1693115234375+0.35,
            2: 2 * 0.1693115234375+0.35,
        }
    elif seqInd == 3:
        expToGuess = {
            1: 0.1693115234375+0.50,
        }
    elif seqInd == 31:
        expToGuess = {
            1: 0.1693115234375 + 0.4,
            2: 2 * 0.1693115234375+ 0.4,
            3: 3 * 0.1693115234375+ 0.4,
            4: 4 * 0.1693115234375+ 0.4
        }  
    else:
        expToGuess = {
            1: 0.1693115234375,
            2: 2 * 0.1693115234375,
            3: 3 * 0.1693115234375,
            4: 4 * 0.1693115234375
        }
    totRA = expToGuess[expInd]
    totDEC = 0

    h = deepcopy(seq.exposures[0].ccds[ccdInd].header)
    for key in ['CRVAL1', 'CENRA1', 'COR1RA1', 'COR2RA1', 'COR3RA1', 'COR4RA1']:
        h[key] += totRA
    
    for key in ['CRVAL2', 'CENDEC1', 'COR1DEC1', 'COR2DEC1', 'COR3DEC1', 'COR4DEC1']:
        h[key] += totDEC

    raBufferPix = 900
    decBufferPix = 100
    raBuffer = raBufferPix * 0.263 / 60 / 60
    decBuffer = decBufferPix * 0.263 / 60 / 60
        
    maxRA = max([h[x] for x in ['COR{}RA1'.format(y) for y in range(1,5)]])
    minRA = min([h[x] for x in ['COR{}RA1'.format(y) for y in range(1,5)]])
    maxDec = max([h[x] for x in ['COR{}DEC1'.format(y) for y in range(1,5)]])
    minDec = min([h[x] for x in ['COR{}DEC1'.format(y) for y in range(1,5)]])

    mask = (reg[:,1] > minRA - raBuffer) * (reg[:,1] < maxRA + raBuffer) *\
        (reg[:,2] > minDec - decBuffer) * (reg[:,2] < maxDec + decBuffer)
    clip = reg[mask]

    wcs = WCS(h)
    pix = wcs.all_world2pix(np.array([clip[:,1], clip[:,2]]).T, 1)
    
    orig = ccd.image
    n,m = orig.shape
    ext = np.zeros((n+2*decBufferPix, m+2*raBufferPix))
    base = np.zeros((n+2*decBufferPix, m+2*raBufferPix))
    base2 = np.zeros((n+2*decBufferPix, m+2*raBufferPix))
    kernel = np.zeros((n+2*decBufferPix, m+2*raBufferPix))
    xcent = kernel.shape[0] // 2 + decBufferPix
    ycent = kernel.shape[1] // 2 + raBufferPix
    xcent=0
    ycent=0
    kernel[xcent+decBufferPix:xcent+20+decBufferPix, ycent+raBufferPix:ycent+800+raBufferPix] = 1
    for x,y in pix.astype('int'):
        if 0 <= y+decBufferPix < n+2*decBufferPix and 0 <= x+raBufferPix < m+2*raBufferPix :
            base[y+decBufferPix,x+raBufferPix] = 1
        
    ext[decBufferPix:-decBufferPix, raBufferPix:-raBufferPix] = orig
    
    res = conv(base, kernel)
    res = np.roll(np.roll(res, -raBufferPix, axis=1), -decBufferPix, axis=0)
    res = res[decBufferPix:-decBufferPix,raBufferPix:-raBufferPix]
    
    normed = np.maximum(0,np.minimum(orig,1000))
    corr = correlate(normed, res, mode='same')
    mm = np.unravel_index(corr.argmax(), corr.shape)
    deltax = mm[1] - corr.shape[1] // 2
    deltay = mm[0] - corr.shape[0] // 2
    return totRA + deltax * 0.263 / 60 / 60, totDEC + deltay * 0.263 / 60 / 60

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-seq", help="sequence", type=int)
    parser.add_argument("-surv", help="survey", type=str, default='core')

    args = parser.parse_args()
    if args.surv != 'core':
        raise NotImplementedError()

    table = Table(names=['seq', 'exp', 'ccd', 'ra', 'dec'], dtype=('i4', 'i4', 'i4', 'f4', 'f4'))
    surv = Survey.getCoreSurvey()
    for expInd in range(1,len(surv.sequences[args.seq])):
        for ccdInd in range(61):
            totRA, totDEC = fastCorrect(surv, args.seq, expInd, ccdInd)
            table.add_row([int(surv.sequences[args.seq].seconds), expInd, ccdInd, totRA, totDEC])
    
    table.write(f'{registration_dir}corrections_{args.surv}_{args.seq}.csv', overwrite=True)