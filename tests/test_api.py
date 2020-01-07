import unittest
from startrail.api import Box, Exposure, StarTrailExposure, StaticExposure, Sequence, Survey, \
                          Registration

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.surv = Survey.getCoreSurvey()

    def test_box(self):
        box = Box([0,0,1,1],[0,1,1,0])
        eps = 1e-4
        for i in [eps, 1 - eps]:
            for j in [eps, 1 - eps]:
                assert box.contains(i, j)
        assert box.contains(0.5, 0.5)
    
    def test_exposure(self):
        fname = '../data/c4d_190528_030041_opi_r_v1.fits.fz'
        tracking = 'Track', 
        exptime = 0.1, 
        seconds = 100
        band = 'r'
        centRA = 239.99
        centDEC = -40.1
        exp = Exposure(fname, tracking, exptime, seconds, band, centRA, centDEC)
    
        assert exp.band == band
        assert exp.contains(centRA, centDEC)
        assert type(exp.seconds) == int
    
        for cls in [StarTrailExposure, StaticExposure]:
            assert cls(fname, tracking, exptime, seconds, band, centRA, centDEC)

    def test_ccd(self):
        ccd = self.surv.sequences[1].exposures[1].ccds[20]
        assert ccd.image.shape == (1995, 3989)
        assert ccd.header['AVSKY'] == 451.1685

    
    def test_sequence(self):
        for seq in self.surv.sequences:
            assert seq[0].seconds < seq[1].seconds
    
        assert self.surv.findSeq(82658).contains(227.65, -17)
    
    def test_survey(self):
        cls = Survey
        methodsToTest = ['getCoreSurvey', 'getTargetSurvey', 'getEngineeringSurvey', 'getAuxiliarySurvey']
        [getattr(cls, x) for x in methodsToTest]
        
        assert self.surv.findSeq('82658.819882') is not None
        assert self.surv.findSeq(82658.819882) is not None
        assert self.surv.findSeq(82658) is not None
        assert self.surv.findSeq(82658) == self.surv.findSeq('82658.819882')
    
    def test_registration(self):
        reg = Registration(82658)
        assert len(reg.data) == 5450
        assert reg.columns[3] == 'phot_rp_mean_flux'
        sources, cols = reg.getSourcesIn(self.surv.sequences[0].exposures[0].ccds[2].polygon)
        assert len(sources) == 61
        assert len(cols) == 5
