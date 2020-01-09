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
RERUN = set([(1, 2, 30),
 (1, 2, 33),
 (7, 3, 19),
 (7, 4, 38),
 (13, 2, 19),
 (22, 3, 49),
 (28, 1, 0),
 (28, 1, 5),
 (28, 1, 8),
 (28, 2, 5),
 (28, 3, 4),
 (28, 3, 7),
 (28, 4, 0),
 (28, 4, 17),
 (35, 2, 44),
 (36, 2, 25),
 (36, 2, 31),
 (36, 2, 32),
 (36, 2, 33),
 (36, 2, 39),
 (36, 2, 45),
 (36, 2, 57),
 (36, 3, 24),
 (36, 3, 31),
 (36, 3, 32),
 (36, 3, 38),
 (36, 4, 24),
 (36, 4, 31),
 (36, 4, 32),
 (36, 4, 38),
 (37, 1, 7),
 (37, 1, 9),
 (37, 2, 14),
 (37, 3, 8),
 (37, 3, 60),
 (37, 4, 13),
 (37, 4, 17),
 (38, 1, 6),
 (38, 1, 14),
 (38, 1, 18),
 (38, 2, 54),
 (38, 3, 5),
 (38, 3, 6),
 (38, 3, 13),
 (38, 4, 53),
 (38, 4, 55),
 (40, 4, 37),
 (40, 4, 44),
 (41, 1, 10),
 (41, 3, 9),
 (42, 1, 3),
 (42, 1, 7),
 (42, 1, 33),
 (42, 2, 0),
 (42, 2, 6),
 (42, 2, 19),
 (42, 2, 43),
 (42, 2, 56),
 (42, 3, 4),
 (42, 3, 16),
 (42, 3, 32),
 (42, 3, 39),
 (43, 1, 12),
 (43, 1, 16),
 (43, 1, 36),
 (43, 1, 37),
 (43, 1, 39),
 (43, 1, 40),
 (43, 1, 51),
 (44, 4, 22),
 (44, 4, 24),
 (45, 1, 23),
 (45, 3, 22),
 (45, 3, 30),
 (45, 3, 33),
 (45, 4, 42),
 (45, 4, 59),
 (46, 1, 24),
 (46, 2, 18),
 (46, 2, 37),
 (46, 2, 45),
 (46, 4, 14),
 (46, 4, 16),
 (46, 4, 44),
 (46, 4, 54),
 (47, 2, 1),
 (47, 2, 17),
 (47, 2, 31),
 (47, 2, 58),
 (47, 3, 35),
 (47, 4, 57),
 (48, 1, 20),
 (48, 1, 57),
 (48, 2, 39),
 (48, 2, 60),
 (48, 3, 7),
 (48, 3, 16),
 (48, 3, 19),
 (48, 3, 42),
 (48, 3, 56),
 (48, 4, 5),
 (48, 4, 21),
 (48, 4, 38),
 (48, 4, 47),
 (49, 1, 21),
 (49, 1, 45),
 (49, 2, 12),
 (49, 4, 28),
 (49, 4, 32),
 (50, 1, 23),
 (50, 2, 9),
 (50, 3, 8),
 (50, 3, 22),
 (50, 3, 59),
 (50, 4, 6),
 (50, 4, 47),
 (51, 3, 4),
 (51, 3, 44),
 (51, 4, 1),
 (51, 4, 4),
 (51, 4, 5),
 (51, 4, 11),
 (51, 4, 22),
 (51, 4, 38),
 (51, 4, 59),
 (55, 1, 37),
 (55, 3, 36),
 (56, 1, 58),
 (56, 2, 44),
 (56, 2, 49),
 (56, 2, 54),
 (56, 3, 18),
 (56, 3, 32),
 (56, 3, 56),
 (57, 1, 36),
 (57, 1, 37),
 (57, 1, 45),
 (57, 1, 46),
 (57, 1, 49),
 (57, 1, 55),
 (57, 1, 56),
 (57, 2, 15),
 (57, 3, 44),
 (57, 3, 45),
 (57, 3, 55),
 (58, 3, 6),
 (58, 3, 55),
 (60, 2, 24),
 (60, 2, 32),
 (60, 2, 39),
 (60, 2, 40),
 (61, 1, 26),
 (61, 3, 25),
 (63, 1, 0),
 (63, 1, 2),
 (63, 1, 4),
 (63, 1, 39),
 (63, 1, 45),
 (63, 1, 49),
 (63, 1, 52),
 (63, 2, 11),
 (63, 2, 20),
 (63, 2, 21),
 (63, 2, 25),
 (63, 2, 59),
 (66, 3, 54),
 (67, 1, 15),
 (67, 1, 48),
 (67, 2, 25),
 (67, 2, 27),
 (67, 3, 14),
 (67, 3, 32),
 (67, 3, 34),
 (67, 3, 47),
 (67, 3, 52),
 (67, 3, 53),
 (67, 3, 55),
 (69, 1, 50),
 (69, 1, 52),
 (69, 2, 34),
 (69, 2, 39),
 (69, 2, 42),
 (69, 2, 48),
 (70, 1, 12),
 (70, 1, 15),
 (70, 1, 32),
 (70, 1, 59),
 (70, 1, 60),
 (70, 2, 8),
 (70, 2, 16),
 (70, 2, 40),
 (70, 3, 2),
 (70, 3, 4),
 (70, 3, 14),
 (70, 3, 31),
 (70, 3, 57)])


GUESS_RA_MAP = {
(1,2): 0.8175033330917358,
(7,3): 0.5994001030921936,
(7,4): 0.7770400047302246,
(13,2): 0.4327186048030853,
(22,3): 0.569739580154419,
(28,1): 0.23440402746200562,
(28,2): 0.3989669382572174,
(28,3): 0.5703970789909363,
(28,4): 0.7443110942840576,
(35,2): 0.38304081559181213,
(36,2): 0.38662055134773254,
(36,3): 0.5609729290008545,
(36,4): 0.7326952815055847,
(37,1): 0.25434818863868713,
(37,2): 0.42687416076660156,
(37,3): 0.5985965132713318,
(37,4): 0.7709763646125793,
(38,1): 0.21372929215431213,
(38,2): 0.389835000038147,
(38,3): 0.5641142725944519,
(38,4): 0.736348032951355,
(40,4): 0.7310880422592163,
(41,1): 0.2203773558139801,
(41,3): 0.5720043182373047,
(42,1): 0.24353596568107605,
(42,2): 0.41986083984375,
(42,3): 0.5957473516464233,
(43,1): 0.2121220827102661,
(44,4): 0.7456260919570923,
(45,1): 0.2176012396812439,
(45,3): 0.5718581676483154,
(45,4): 0.745918333530426,
(46,1): 0.23411180078983307,
(46,2): 0.4064185917377472,
(46,4): 0.7566574811935425,
(47,2): 0.3782191574573517,
(47,3): 0.5522792935371399,
(47,4): 0.7301383018493652,
(48,1): 0.21394847333431244,
(48,2): 0.39136916399002075,
(48,3): 0.5668173432350159,
(48,4): 0.7444571852684021,
(49,1): 0.22286124527454376,
(49,2): 0.4026927649974823,
(49,4): 0.7484022378921509,
(50,1): 0.21292568743228912,
(50,2): 0.3892505466938019,
(50,3): 0.5636759400367737,
(50,4): 0.7357636094093323,
(51,3): 0.5656484365463257,
(51,4): 0.7372246980667114,
(55,1): 0.21964679658412933,
(55,3): 0.5683515071868896,
(56,1): 0.21380235254764557,
(56,2): 0.3861091732978821,
(56,3): 0.5635298490524292,
(57,1): 0.22622179985046387,
(57,2): 0.39100387692451477,
(57,3): 0.566305935382843,
(58,3): 0.5747073292732239,
(60,2): 0.4099983274936676,
(61,1): 0.22023124992847443,
(61,3): 0.5685706734657288,
(63,1): 0.2737809717655182,
(63,2): 0.4482794404029846,
(66,3): 0.5622148513793945,
(67,1): 0.2325776368379593,
(67,2): 0.40612637996673584,
(67,3): 0.5843507051467896,
(69,1): 0.21489819884300232,
(69,2): 0.38778942823410034,
(70,1): 0.21453291177749634,
(70,2): 0.389104425907135,
(70,3): 0.5635298490524292,
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
    for seqInd, expInd, ccdInd in RERUN:
        deltaRA, deltaDEC = fastCorrect(surv, seqInd, expInd, ccdInd)
        table.add_row([seqInd, expInd, ccdInd, deltaRA, deltaDEC])
    table.write(f'{registration_dir}corrections_core_individual.csv', overwrite=True)