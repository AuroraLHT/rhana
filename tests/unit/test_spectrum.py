import json
import unittest
from pathlib import Path

from scipy.sparse.construct import random
from rhana.spectrum.spectrum import Spectrum
from rhana.spectrum.model import FuncSpectrumModelConfig, SpectrumModel, SpectrumModelConfig, maximum_amplitude
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
        self.spec.normalize()
        self.spec.remove_background()
        self.spec.savgol()
        self.spec.smooth()
        self.spec.clip(0, 4)
        
    def testOperation(self):
        self.spec.crop(1, len(self.spec.ws) -2, inplace=False)
        self.spec.normalize(inplace=False)
        self.spec.remove_background(inplace=False)
        self.spec.savgol(inplace=False)
        self.spec.smooth(inplace=False)
        self.spec.clip(0, 4, inplace=False)

    def testSave(self):
        preprocessed = self.spec.remove_background(inplace=False).normalize(False).smooth(4, False)
        peaks, peaksinfo = preprocessed.fit_spectrum_peaks(height=0.01, threshold=0, prominence=0.01)
        bg_mask = preprocessed.spec <= 0.1

        config = SpectrumModelConfig(
            height = {"min":0},
            sigma = {"min":1e-6,"max":0.1},
            center = {"min":preprocessed.ws.min(), "max":preprocessed.ws.max()},
            amplitude = {"min":0, "max": maximum_amplitude(self.spec)},
            type = "VoigtModel",
            # type = "LorentzianModel",
            # type="GaussianModel",
            poly_n = 7,
            poly_zero_init=True,
            peak_window = 45,
            add_vogit_bg=False,
            vogit_bg_amp_ratio = 0.01,
            center_search_width=1,
        )

        sm = SpectrumModel.from_peak_finding(peaks, peaksinfo, self.spec, config, bg_mask, by="guess")
        output = sm.fit(self.spec, method="leastsq")

        self.spec.sm = sm

        save_path = Path(__file__).parent.parent / Path('tmp')
        save_path.mkdir(exist_ok=True)

        self.spec.save(save_path)
        Spectrum.load(save_path)


class TestSpectrumModelConfig(unittest.TestCase):
    def setUp(self):
        with (Path(__file__).parent.parent / "fixtures/single_spec_data.json").open("r") as f:
            tmp = json.load(f)
            spec = np.array(tmp['spec'])
            ws = np.array(tmp['ws'])

        self.spec = Spectrum(spec=spec, ws=ws)

    def testSpectrumModelConfig(self):
        config = SpectrumModelConfig(
            height = {"min":0},
            sigma = {"min":1e-6,"max":0.1},
            center = {"min":self.spec.ws.min(), "max":self.spec.ws.max()},
            amplitude = {"min":0, "max": maximum_amplitude(self.spec)},
            type = "VoigtModel",
            # type = "LorentzianModel",
            # type="GaussianModel",
            poly_n = 7,
            poly_zero_init=True,
            peak_window = 45,
            add_vogit_bg=False,
            vogit_bg_amp_ratio = 0.01,
            center_search_width=1,
        )

        for k in config.__dict__.keys():
            self.assertIsNotNone(getattr(config, k))


    def testFuncSpectrumModelConfig(self):
        func_config = FuncSpectrumModelConfig(
            height = {"min":0},
            sigma = {"min":1e-6,"max":0.1},
            center = {"min":self.spec.ws.min(), "max":self.spec.ws.max()},
            amplitude = {"min":0, "max": maximum_amplitude(self.spec)},
            type = lambda : np.random.choice(["VoigtModel", "LorentzianModel", "GaussianModel"]),
            poly_n = lambda : np.random.randint(0, 7),
            poly_zero_init = False,
            peak_window = lambda : np.random.randint(30, 50),
            add_vogit_bg = False,
            vogit_bg_amp_ratio = 0.01,
            center_search_width = lambda : np.random.rand()*2,
        )

        for i in range(5):
            config = func_config.evaluate()
            for k in config.__dict__.keys():
                self.assertIsNotNone(getattr(config, k))

