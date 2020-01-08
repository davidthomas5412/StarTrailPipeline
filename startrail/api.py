import os
from math import sqrt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib.path import Path
from startrail.paths import data_dir, registration_dir, summary_table

def timeToSeconds(time):
    h = int(time[:2])
    m = int(time[3:5])
    s = float(time[6:])
    t = 3600 * h + 60 * m + s
    # so that survey is ordered correctly
    if t < 40000:
        t += 24 * 3600
    return t

def lazy_property(fn):
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property

class Survey:
    data = Table.read(summary_table)
    data['seconds'] = [timeToSeconds(x) for x in data['TIME-OBS']]
    data.sort('seconds')
    science_mask = np.array(['2019-07' in x for x in data['DATE-OBS']])
    tracking_mask = data['TELSTAT'] == 'Track'
    extra_mask = np.zeros(len(data), dtype='bool')
    extra_mask[395:] = True
    target_mask = np.zeros(len(data), dtype='bool')

    # MAXI J1820+070
    for i in range(390,395):
        target_mask[i] = True

    # V4641 Sgr
    for i in range(370,375):
        target_mask[i] = True

    def __init__(self, name, data):
        sequences = []
        for row in data:
            inp = [row[x] for x in ('fname', 'EXPTIME', 'TIME-OBS', 'seconds', 'BAND', 'CENTRA', 'CENTDEC')]
            inp[0] =  data_dir + inp[0]
            if row['TELSTAT'] == 'Track':
                exp = StaticExposure(*inp)
                seq = Sequence(exp)
                sequences.append(seq)
            else:
                exp = StarTrailExposure(*inp)
                seq.addExposure(exp)

        self.name = name
        self.sequences = sequences
        self.secondsToSeq = dict()
        for seq in self.sequences:
            self.secondsToSeq[int(seq.seconds)] = seq

    def contains(self, x, y):
        for seq in self.sequences:
            if seq.contains(x, y):
                return True
        return False

    def findSeq(self, seconds):
        sec = int(float(seconds)) # accepts int, float, or str
        return self.secondsToSeq[sec]

    def __len__(self):
        return len(self.sequences)

    @staticmethod
    def getCoreSurvey():
        mask = Survey.science_mask * ~Survey.extra_mask * ~Survey.target_mask
        return Survey('core', Survey.data[mask])

    @staticmethod
    def getTargetSurvey():
        mask = Survey.target_mask
        return Survey('target', Survey.data[mask])

    @staticmethod
    def getEngineeringSurvey():
        mask = ~Survey.science_mask
        return Survey('engineering', Survey.data[mask])

    @staticmethod
    def getAuxiliarySurvey():
        mask = Survey.extra_mask
        return Survey('auxiliary', Survey.data[mask])

class Sequence:
    def __init__(self, keyExposure):
        self.exposures = [keyExposure]
        self.seconds = keyExposure.seconds # we use this as unique identifier
        self.band = keyExposure.band
        self.centRA = keyExposure.centRA
        self.centDEC = keyExposure.centDEC
        self.registration = Registration(self.seconds)

    def addExposure(self, exposure):
        self.exposures.append(exposure)
        self.exposures.sort(key=lambda exp: exp.seconds)

    def contains(self, x, y):
        cutoff = 2
        r = sqrt((x - self.centRA) ** 2 + (y - self.centDEC) ** 2)
        if r > cutoff:
            return False

        for exp in self.exposures:
            if exp.contains(x,y):
                return True
        return False

    def __getitem__(self, n):
        return self.exposures[n]

    def __len__(self):
        return len(self.exposures)

class Exposure:
    def __init__(self, fname, tracking, exptime, seconds, band, centRA, centDEC):
        self.fname = fname
        self.tracking = tracking
        self.exptime = exptime
        self.seconds = seconds
        self.band = band
        self.centRA = centRA
        self.centDEC = centDEC

    @lazy_property
    def header(self):
        return fits.open(self.fname)[0].header

    @lazy_property
    def ccds(self):
        hdus = fits.open(self.fname)
        return [CCD(hdu) for hdu in hdus[1:]]

    def contains(self, x, y):
        cutoff = 1.2
        r = sqrt((x - self.centRA) ** 2 + (y - self.centDEC) ** 2)
        if r > cutoff:
            return False
        
        return any((ccd.contains(x, y) for ccd in self.ccds))

class StarTrailExposure(Exposure):
    def __init__(self, fname, tracking, exptime, timeObs, band, centRA, centDEC):
        super(StarTrailExposure, self).__init__(fname, tracking, exptime, timeObs, band, centRA, centDEC)

class StaticExposure(Exposure):
    def __init__(self, fname, tracking, exptime, timeObs, band, centRA, centDEC):
        super(StaticExposure, self).__init__(fname, tracking, exptime, timeObs, band, centRA, centDEC)

class Box:
    def __init__(self, cornersX, cornersY):
        if len(cornersX) != 4 or len(cornersY) !=4:
            raise RuntimeError('Need 4 corners for Box.')
        self.cornersX = cornersX
        self.cornersY = cornersY
        self.polygon = Path(np.vstack([cornersX, cornersY]).T)

    def contains(self, x, y):
        return self.polygon.contains_point((x,y))

class CCD(Box):
    def __init__(self, hdu):
        cornersX = [hdu.header['COR{}RA1'.format(i)] for i in [1,2,4,3]]
        cornersY = [hdu.header['COR{}DEC1'.format(i)] for i in [1,2,4,3]]
        super(CCD, self).__init__(cornersX, cornersY)

        self.wcs = None
        self.hdu = hdu

    @lazy_property
    def image(self):
        return self.hdu.data.T

    @lazy_property
    def header(self):
        return self.hdu.header

    def plot(self):
        # TODO: implement
        raise NotImplementedError()

    def pix2World(self, pixX, pixY):
        if self.wcs is None:
            self.wcs = WCS(self.header)
        return self.wcs.all_pix2world(np.array([pixX, pixY]).T, 1)

    def world2Pix(self, ra, dec):
        if self.wcs is None:
            self.wcs = WCS(self.header)
        return self.wcs.all_world2pix(np.array([ra, dec]).T, 1)


class Registration:
    def __init__(self, seconds):
        self.key = int(float(seconds))
        self.fname = os.path.join(registration_dir, f'merged_{self.key}.csv')

    def getSourcesIn(self, polygon):
        if isinstance(polygon, Box):
            polygon = polygon.polygon
        mask = [polygon.contains_point((x[1], x[2])) for x in self.data]
        return self.data[mask], self.columns

    @lazy_property
    def data(self):
        return np.loadtxt(self.fname, delimiter=',')
    
    @lazy_property
    def columns(self):
        with open(self.fname, 'r') as r:
            columns = r.readline()[2:].split(',')
        return columns
