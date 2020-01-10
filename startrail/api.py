from math import sqrt
import numpy as np
from os.path import join
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib.path import Path
from startrail.paths import data_dir, registration_dir, summary_table, adjust_table

def time_to_seconds(time):
    h = int(time[:2])
    m = int(time[3:5])
    s = float(time[6:])
    t = 3600 * h + 60 * m + s
    # so that survey is ordered correctly despite being through midnight
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
    valid_table = Table.read(valid_table)
    adjust_table = Table.read(adjust_table)
    data = Table.read(summary_table)
    data['seconds'] = [time_to_seconds(x) for x in data['TIME-OBS']]
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
        for seq_ind, row in enumerate(data):
            inp = [row[x] for x in ('fname', 'EXPTIME', 'TIME-OBS', 'seconds', 'BAND', 'CENTRA', 'CENTDEC')]
            inp[0] =  join(data_dir, inp[0])
            if row['TELSTAT'] == 'Track':
                exp_ind = 0
                exp = StaticExposure(exp_ind, *inp)
                seq = Sequence(seq_ind, exp)
                sequences.append(seq)
                
            else:
                adj_sub = Survey.adjust_subtable(seq_ind, exp_ind)
                val_sub = Survey.valid_subtable(seq_ind, exp_ind)
                exp = StarTrailExposure(exp_ind, *inp, adjust_table=adj_sub, valid_table=val_sub)
                seq.add_exposure(exp)
            exp_ind += 1

        self.name = name
        self.sequences = sequences
        self.seconds_to_seq = dict()
        for seq in self.sequences:
            self.seconds_to_seq[int(seq.seconds)] = seq

    @staticmethod
    def valid_subtable(seq_ind, exp_ind):
        sub = valid_table[(valid_table['seq'] == seq_ind) * (valid_table['exp'] == exp_ind)]
        return sub[['ccd_ind', 'valid']]

    @staticmethod
    def adjust_subtable(seq_ind, exp_ind):
        sub = adjust_table[(adjust_table['seq'] == seq_ind) * (adjust_table['exp'] == exp_ind)]
        return sub[['ccd_ind', 'ra', 'dec']]

    def contains(self, x, y):
        for seq in self.sequences:
            if seq.contains(x, y):
                return True
        return False

    def find_seq(self, seconds):
        sec = int(float(seconds)) # accepts int, float, or str
        return self.seconds_to_seq[sec]

    def __len__(self):
        return len(self.sequences)

    @staticmethod
    def get_core_survey():
        mask = Survey.science_mask * ~Survey.extra_mask * ~Survey.target_mask
        return Survey('core', Survey.data[mask])

    @staticmethod
    def get_target_survey():
        # mask = Survey.target_mask
        # return Survey('target', Survey.data[mask])
        raise NotImplementedError()

    @staticmethod
    def get_engineering_survey():
        # mask = ~Survey.science_mask
        # return Survey('engineering', Survey.data[mask])
        raise NotImplementedError()

    @staticmethod
    def get_auxiliary_survey():
        # mask = Survey.extra_mask
        # return Survey('auxiliary', Survey.data[mask])
        raise NotImplementedError()

class Sequence:
    def __init__(self, index, keyExposure):
        self.index = index
        self.exposures = [keyExposure]
        self.seconds = keyExposure.seconds # we use this as unique identifier
        self.band = keyExposure.band
        self.ra = keyExposure.ra
        self.dec = keyExposure.dec
        self.registration = Registration(self.seconds)

    def add_exposure(self, exposure):
        self.exposures.append(exposure)
        self.exposures.sort(key=lambda exp: exp.seconds)

    def extract(self, exp_ind, ccd_ind, cutoff):
        # TODO: return iterator
        raise NotImplementedError()

    def contains(self, x, y):
        cutoff = 2
        r = sqrt((x - self.ra) ** 2 + (y - self.dec) ** 2)
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
    def __init__(self, index, fname, tracking, exptime, seconds, band, ra, dec):
        self.index = index
        self.fname = fname
        self.tracking = tracking
        self.exptime = exptime
        self.seconds = seconds
        self.band = band
        self.ra = ra
        self.dec = dec

    @lazy_property
    def header(self):
        return fits.open(self.fname)[0].header

    @lazy_property
    def ccds(self):
        hdus = fits.open(self.fname)
        return [CCD(i,hdu) for i,hdu in enumerate(hdus[1:])]

    def contains(self, x, y):
        cutoff = 1.2
        r = sqrt((x - self.ra) ** 2 + (y - self.dec) ** 2)
        if r > cutoff:
            return False
        
        return any((ccd.contains(x, y) for ccd in self.ccds))

class StarTrailExposure(Exposure):

    def __init__(self, index, fname, tracking, exptime, time_obs, band, ra, dec, adjust_table=None, valid_table=None):
        if adjust_table and len(adjust_table) > 0:
            self.adjust_table.sort('ccd_ind')
            ra, dec = StarTrailExposure.adjust_center(ra, dec)
        super(StarTrailExposure, self).__init__(index, fname, tracking, exptime, time_obs, band, ra, dec)
        if valid_table:
            self.valid_table = valid_table
            self.valid_table.sort('ccd_ind')

    @lazy_property
    def header(self):
        return StarTrailExposure.adjust_header(fits.open(self.fname)[0].header)

    @lazy_property
    def ccds(self):
        hdus = fits.open(self.fname)
        hdus = StarTrailExposure.adjust_ccds(hdus)
        return [CCD(i,hdu,valid) for i,hdu,valid in zip(range(61), hdus[1:], self.valid_table['valid'])]

    @lazy_property
    def valid_ccds(self):
        hdus = fits.open(self.fname)
        hdus = StarTrailExposure.adjust_ccds(hdus)
        return [CCD(i,hdu,valid) for i,hdu,valid in zip(range(61), hdus[1:], self.valid_table['valid']) if valid]

    @staticmethod
    def adjust_ccds(hdus):
        if not self.adjust_table:
            return hdus 

        for ccd_ind, hdu in enumerate(hdus[1:]):
            row = self.adjust_table[self.adjust_table['ccd_ind'] == ccd_ind]
            d_ra = row['ra']
            d_dec = row['dec']
            for key in ['CRVAL1', 'CENRA1', 'COR1RA1', 'COR2RA1', 'COR3RA1', 'COR4RA1']:
                hdu[key] += d_ra
    
            for key in ['CRVAL2', 'CENDEC1', 'COR1DEC1', 'COR2DEC1', 'COR3DEC1', 'COR4DEC1']:
                hdu[key] += d_dec
        return hdus

    @staticmethod
    def adjust_center(ra, dec):
        row = self.adjust_table[self.adjust_table['ccd_ind'] == 30]
        ra += row['ra'] 
        dec += row['dec'] 
        return ra, dec

    @staticmethod
    def adjust_header(header):
        if not self.adjust_table:
            return header
            
        row = self.adjust_table[self.adjust_table['ccd_ind'] == 30]
        d_ra = row['ra'] 
        d_dec = row['dec'] 
        for key in ['CENTRA', 'CORN1RA', 'CORN2RA', 'CORN3RA', 'CORN4RA']:
            header[key] += d_ra
        for key in ['CENTDEC', 'CORN1DEC', 'CORN2DEC', 'CORN3DEC', 'CORN4DEC']:
            header[key] += d_dec
        return header


class StaticExposure(Exposure):
    def __init__(self, index, fname, tracking, exptime, timeObs, band, centRA, centDEC):
        super(StaticExposure, self).__init__(index, fname, tracking, exptime, timeObs, band, centRA, centDEC)

class Box:
    def __init__(self, corners_x, corners_y):
        if len(corners_x) != 4 or len(corners_y) !=4:
            raise RuntimeError('Need 4 corners for Box.')
        self.corners_x = corners_x
        self.corners_y = corners_y
        self.polygon = Path(np.vstack([corners_x, corners_y]).T)

    def contains(self, x, y):
        return self.polygon.contains_point((x,y))

class CCD(Box):
    def __init__(self, ccd_ind, hdu, valid):
        corners_x = [hdu.header['COR{}RA1'.format(i)] for i in [1,2,4,3]]
        corners_x = [hdu.header['COR{}DEC1'.format(i)] for i in [1,2,4,3]]
        super(CCD, self).__init__(corners_x, corners_x)
        self.ccd_ind = ccd_ind
        self.wcs = None
        self.hdu = hdu
        self.valid = valid

    @lazy_property
    def image(self):
        return self.hdu.data.T

    @lazy_property
    def header(self):
        return self.hdu.header

    def plot(self, vmin=0, vmax=100, cmap='gray', origin='lower'):
        fig, ax = plt.subplots()
        ax.set_title(f'CCD: {ccd_ind}')
        ax.imshow(self.image.T, vmin=vmin, vmax=vmax, cmap=cmap, origin=origin)
        return fig, ax

    def pix_to_world(self, pix_x, pix_y):
        if self.wcs is None:
            self.wcs = WCS(self.header)
        return self.wcs.all_pix2world(np.array([pix_x, pix_y]).T, 1)

    def world_to_pix(self, ra, dec):
        if self.wcs is None:
            self.wcs = WCS(self.header)
        return self.wcs.all_world2pix(np.array([ra, dec]).T, 1)

class Registration:
    def __init__(self, seconds):
        self.key = int(float(seconds))
        self.fname = join(registration_dir, f'merged_{self.key}_500.csv') # 500 query mean flux cutoff

    def get_sources_in(self, polygon):
        if isinstance(polygon, Box):
            polygon = polygon.polygon
        mask = [polygon.contains_point((x[1], x[2])) for x in self.data]
        return self.data[mask], self.columns

    @lazyProperty
    def data(self):
        return np.loadtxt(self.fname, delimiter=',')
    
    @lazyProperty
    def columns(self):
        with open(self.fname, 'r') as r:
            columns = r.readline()[2:].split(',')
        return columns

class Trail:
    def __init__(self, gaiaId, ra, dec, flux, img, start, end):
        self.id = gaiaId
        self.ra = ra
        self.dec = dec
        self.flux = flux
        self.image = image
        self.start = start
        self.end = end

    def plot(self, vmin=0, vmax=100, cmap='gray', origin='lower'):
        fig, ax = plt.subplots()
        ax.set_title(f'CCD: {ccd_ind}')
        ax.imshow(self.image.T, vmin=vmin, vmax=vmax, cmap=cmap, origin=origin)
        return fig, ax

