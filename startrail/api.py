from math import sqrt
import numpy as np
from copy import deepcopy
from os.path import join
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib.path import Path
from startrail import paths
import matplotlib.pyplot as plt

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
    valid_table = Table.read(paths.valid_table)
    adjust_table = Table.read(paths.adjust_table)
    data = Table.read(paths.summary_table)
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
        for row in data:
            inp = [row[x] for x in ('fname', 'EXPTIME', 'TIME-OBS', 'seconds', 'BAND', 'CENTRA', 'CENTDEC')]
            inp[0] =  join(paths.data_dir, inp[0])
            if row['TELSTAT'] == 'Track':
                exp_ind = 0
                exp = StaticExposure(exp_ind, *inp)
                seq_ind = len(sequences)
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
        sub = Survey.valid_table[(Survey.valid_table['seq'] == seq_ind) * (Survey.valid_table['exp'] == exp_ind)]
        return sub[['ccd', 'valid']]

    @staticmethod
    def adjust_subtable(seq_ind, exp_ind):
        sub = Survey.adjust_table[(Survey.adjust_table['seq'] == seq_ind) * (Survey.adjust_table['exp'] == exp_ind)]
        return sub

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
        self.registration = Registration(self.index)

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
        return [CCD(i,hdu,True) for i,hdu in enumerate(hdus[1:])]

    def contains(self, x, y):
        cutoff = 1.2
        r = sqrt((x - self.ra) ** 2 + (y - self.dec) ** 2)
        if r > cutoff:
            return False
        
        return any((ccd.contains(x, y) for ccd in self.ccds))

class StarTrailExposure(Exposure):
    MID_CCD = 30

    def __init__(self, index, fname, tracking, exptime, time_obs, band, ra, dec, adjust_table=None, valid_table=None):
        self.adjust_table = adjust_table
        self.valid_table = valid_table
        if adjust_table and len(adjust_table) > 0:
            self.adjust_table.sort('ccd')
            idx = np.where(self.adjust_table['ccd'] == StarTrailExposure.MID_CCD)[0][0]
            row = self.adjust_table[idx]
            ra = row['CENTRA'] + row['ra'] 
            dec = row['CENTDEC'] + row['dec'] 
        super(StarTrailExposure, self).__init__(index, fname, tracking, exptime, time_obs, band, ra, dec)
        if valid_table:
            self.valid_table.sort('ccd')

    @lazy_property
    def header(self):
        header = fits.open(self.fname)[0].header
        
        if self.adjust_table:
            idx = np.where(self.adjust_table['ccd'] == StarTrailExposure.MID_CCD)[0][0]
            row = self.adjust_table[idx]
            d_ra = row['ra'] 
            d_dec = row['dec'] 
            for key in ['CENTRA', 'CORN1RA', 'CORN2RA', 'CORN3RA', 'CORN4RA']:
                header[key] = row[key] + d_ra
            for key in ['CENTDEC', 'CORN1DEC', 'CORN2DEC', 'CORN3DEC', 'CORN4DEC']:
                header[key] = row[key] + d_dec
        return header

    @lazy_property
    def __hdus(self):
        hdus = fits.open(self.fname)
        
        if self.adjust_table:
            for ccd_ind, hdu in enumerate(hdus[1:]):
                idx = np.where(self.adjust_table['ccd'] == ccd_ind)[0][0]
                row = self.adjust_table[idx]
                d_ra = row['ra']
                d_dec = row['dec']
                for key in ['CRVAL1', 'CENRA1', 'COR1RA1', 'COR2RA1', 'COR3RA1', 'COR4RA1']:
                    hdu.header[key] = row[key] + d_ra
                for key in ['CRVAL2', 'CENDEC1', 'COR1DEC1', 'COR2DEC1', 'COR3DEC1', 'COR4DEC1']:
                    hdu.header[key] = row[key] + d_dec
                for key in ['PV1_7','PV2_8','PV2_9','CD1_1','PV2_0','PV2_1','PV2_2','PV2_3',\
                'PV2_4','PV2_5','PV2_6','PV2_7','PV1_6','PV2_10','PV1_4','PV1_3','PV1_2','PV1_1',\
                'PV1_0','PV1_9','PV1_8','CD1_2','PV1_5','CD2_1','CD2_2','PV1_10']:
                    hdu.header[key] = row[key]

        return hdus


    @lazy_property
    def ccds(self, onlyValid=False):
        if self.valid_table:
            return [CCD(i,hdu,valid) for i,hdu,valid in zip(range(61), self.__hdus[1:], self.valid_table['valid'])]
        return [CCD(i,hdu,True) for i,hdu in zip(range(61), self.__hdus[1:])]


    @lazy_property
    def valid_ccds(self):
        if self.valid_table:
            return [CCD(i,hdu,valid) for i,hdu,valid in zip(range(61), self.__hdus[1:], self.valid_table['valid']) if valid]
        return [CCD(i,hdu,True) for i,hdu in zip(range(61), self.__hdus[1:])]

class StaticExposure(Exposure):
    def __init__(self, index, fname, tracking, exptime, time_obs, band, ra, dec):
        super(StaticExposure, self).__init__(index, fname, tracking, exptime, time_obs, band, ra, dec)

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
    def __init__(self, index, hdu, valid):
        corners_x = [hdu.header['COR{}RA1'.format(i)] for i in [1,2,4,3]]
        corners_y = [hdu.header['COR{}DEC1'.format(i)] for i in [1,2,4,3]]
        super(CCD, self).__init__(corners_x, corners_y)
        self.index = index
        self.wcs = None
        self.hdu = hdu
        self.valid = valid

    @lazy_property
    def image(self):
        return self.hdu.data

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
    def __init__(self, index):
        self.full_catalog_file = join(paths.registration_dir, f'registration_{index}_500.csv')
        self.sparse_catalog_file = join(paths.registration_dir, f'registration_{index}_3000.csv')

    def get_sources_in(self, polygon, cutoff=3000):
        if isinstance(polygon, Box):
            polygon = polygon.polygon
        if cutoff >= 3000:
            sub = self.sparse_catalog[[self.sparse_catalog.columns[3] >= cutoff]]
        else:
            sub = self.full_catalog[[self.full_catalog.columns[3] >= cutoff]]
        mask = [polygon.contains_point((x[1], x[2])) for x in sub]
        return sub[[mask]]

    def get_sources_around(self, polygon, cutoff=3000):
        if isinstance(polygon, Box):
            polygon = polygon.polygon
        if cutoff >= 3000:
            sub = self.sparse_catalog[[self.sparse_catalog.columns[3] >= cutoff]]
        else:
            sub = self.full_catalog[[self.full_catalog.columns[3] >= cutoff]]

        cpy = deepcopy(polygon)
        # for stars that trail in
        sidereal_buffer = 0.0625
        cpy.vertices[:2,0] -= sidereal_buffer 

        # for trails right on the edge
        edge_buffer = 0.005
        cpy.vertices[1:3,1] -= edge_buffer
        cpy.vertices[0,1] += edge_buffer
        cpy.vertices[3,1] += edge_buffer

        mask = [cpy.contains_point((x[1], x[2])) for x in sub]
        return sub[[mask]]

    @lazy_property
    def full_catalog(self):
        '''
        gaia mean flux > 500
        '''
        return Table.read(self.full_catalog_file)

    @lazy_property
    def sparse_catalog(self):
        '''
        gaia mean flux > 3000
        '''
        return Table.read(self.sparse_catalog_file)
    
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
