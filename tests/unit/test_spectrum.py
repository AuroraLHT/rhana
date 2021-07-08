import json
import unittest
from pathlib import Path
from rhana.spectrum.spectrum import Spectrum
from rhana.spectrum.model import SpectrumModel, SpectrumModelConfig, maximum_amplitude
import numpy as np

class TestSpectrum(unittest.TestCase):
    def setUp(self):
        with (Path(__file__).parent.parent / "fixtures/single_spec_data.json").open("r") as f:
            tmp = json.load(f)
            spec = np.array(tmp['spec'])
            ws = np.array(tmp['ws'])

        self.spec = Spectrum(spec=spec, ws=ws)
    
    def testOperationInplace(self):
        self.spec.crop(1, len(self.spec.ws) -2)
        self.spec.normalization()
        self.spec.remove_background()
        self.spec.savgol()
        self.spec.smooth()
        self.spec.clip(0, 4)
        
    def testOperation(self):
        self.spec.crop(1, len(self.spec.ws) -2, inplace=False)
        self.spec.normalization(inplace=False)
        self.spec.remove_background(inplace=False)
        self.spec.savgol(inplace=False)
        self.spec.smooth(inplace=False)
        self.spec.clip(0, 4, inplace=False)

    def testSave(self):
        preprocessed = self.spec.remove_background(inplace=False).normalization(False).smooth(4, False)
        peaks, peaksinfo = preprocessed.fit_spectrum_peaks(height=0.01, threshold=0, prominence=0.01)
        bg_mask = preprocessed.spec <= 0.1

        config = SpectrumModelConfig(
            height = {"min":0},
            sigma = {"min":1e-6,"max":0.1},
            center = {"min":preprocessed.ws.min(), "max":preprocessed.ws.max()},
            amplitude = {"min":0, "max": maximum_amplitude(self.spec)},
            type = "VoigtModel",
        #     type = "LorentzianModel",
        #     type="GaussianModel",
            poly_n = 7,
            poly_zero_init=True,
            peak_window = 45,
            add_vogit_bg=False,
            vogit_bg_amp_ratio = 0.01,
            center_search_width=1,
        )

        sm = SpectrumModel.from_peaks(peaks, peaksinfo, self.spec, config, bg_mask, by="guess")
        output = sm.fit(self.spec, method="leastsq")

        self.spec.sm = sm

        save_path = Path(__file__).parent.parent / Path('tmp')
        save_path.mkdir(exist_ok=True)

        self.spec.save(save_path)
        Spectrum.load(save_path)
