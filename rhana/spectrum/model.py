from dataclasses import dataclass
import signal

import numpy as np

from lmfit.model import Parameters, Parameter
from lmfit import models as lm_models
from lmfit.model import save_modelresult, save_model

from rhana.utils import _create_figure
from rhana.spectrum.spectrum import Spectrum

def maximum_amplitude(spectrum):
    # intergral by trapezoidal rule
    return float(np.trapz(spectrum.spec, spectrum.ws))

@dataclass
class SpectrumModelConfig:
    """
        all parameter with dict type follow this convention
        {"value":..., "vary":..., "min":..., "max":..., "expr":...}
    """
    height : dict
    sigma : dict
    center : dict
    amplitude : dict
    type : list
    poly_n : int
    peak_window : int
    add_vogit_bg : bool
    vogit_bg_amp_ratio : float
    center_search_width : float

class SpectrumModel:
    """
        Reference 
        1. https://chrisostrouchov.com/post/peak_fit_xrd_python/
        2. https://lmfit.github.io/lmfit-py/examples/documentation/builtinmodels_nistgauss2.html#sphx-glr-examples-documentation-builtinmodels-nistgauss2-py
    """
    
    _model_prefix = {
        "GaussianModel":"G{}_",
        "LorentzianModel":"L{}_",
        "VoigtModel":"V{}_"
    }
    
    def __init__(self, model, sub_models, params,):
        self.sub_models = sub_models # list of lmfit models
        self.model = model
        self.params = params
    
    @classmethod
    def from_peaks(cls, peaks, peaks_info, spec, config, bg_mask, by="guess"):
        composite_model = None
        sub_models = []
        params = None
        
        def _update(model, model_params, params, sub_models, composite_model):
            if isinstance(model_params, dict):
                model_params = model.make_params(**model_params)
            else:
                model_params = model.make_params(**params)
                
            if params is None:
                params = model_params
            else:
                params.update(model_params)
#             display(params)    
            if composite_model is None:
                composite_model = model
            else:
                composite_model = composite_model + model            
             
            sub_models.append(model)
        
            return params, sub_models, composite_model
        
        def _guess_FWHM(spec, peaks, peak_heights, config):
            window = config.peak_window
            wms = peak_heights / 2
            
            above_wm = [ spec.spec[max(p-window,0) : p+window ] < wm for p, wm in zip(peaks, wms) ]
            x = [ spec.ws[max(p-window,0) : p+window ] for p in peaks ]
            
            peak_lefts = []
            peak_rights = []
            for i in range(len(above_wm)):
                above_wm_int = np.where(~above_wm[i])[0]
                if len(above_wm_int)>0:
                    left = above_wm_int.min()
                    right = above_wm_int.max()
                else:
                    left = 0
                    right = len(x[i])-1
                    
                peak_lefts.append(x[i][left])
                peak_rights.append(x[i][right])
                
            peak_widths = np.array(peak_rights) - np.array(peak_lefts)
            return peak_widths
        
        def _default_params_from_peaks(type, prefix, p, p_w, p_h, config):
            if type == "GaussianModel":
                # default guess is horrible!! do not use guess()

                center = p
                
                sigma = p_w / 2.355
                amplitude = float(p_h * (sigma * np.sqrt(2*np.pi)))

                default_params = {
                    f"{prefix}center": center,
                    f"{prefix}amplitude": amplitude,
                    f"{prefix}sigma": sigma
                }
            elif type == "LorentzianModel":
                center = p
                sigma = p_w / 2
                amplitude = float(p_h * (sigma * np.pi))

                default_params = {
                    f"{prefix}center": center,
                    f"{prefix}amplitude": amplitude,
                    f"{prefix}sigma": sigma
                }

            elif type == "VoigtModel":
                center = p
                sigma = p_w / 3.6013
                amplitude = float(p_h * (sigma * np.sqrt(2*np.pi)))

                default_params = {
                    f"{prefix}center": center,
                    f"{prefix}amplitude": amplitude,
                    f"{prefix}sigma": sigma
                }
            else:
                raise NotImplementedError("Unknown type: {type}")

            return default_params        
        
        peak_heights = spec.spec[peaks]
        peak_xs = spec.ws[peaks]
        peak_widths = _guess_FWHM(spec, peaks, peak_heights, config)
        
        for i,(p, p_x, p_w, p_h,) in enumerate(zip(peaks, peak_xs, peak_widths, peak_heights)):
            p_x, p_w, p_h = float(p_x), float(p_w), float(p_h)
            try:
                if isinstance(config.type, str):
                    m_type = config.type
                    prefix = cls._model_prefix[m_type].format(i)
                    model = getattr(lm_models, m_type)(prefix=prefix)
                elif isinstance(config.type, type):
                    m_type = config.type.__name__
                    prefix = cls._model_prefix[m_type].format(i)
                    model = config.type(prefix=prefix)
            except Exception as e:
                raise NotImplemented(f'model {config.type} not implemented yet')
            
            
            center = dict(config.center)
            if config.center_search_width is not None:
                center['min'] = p_x - config.center_search_width / 2
                center['max'] = p_x + config.center_search_width / 2
            
            model.set_param_hint('sigma', **config.sigma)
            model.set_param_hint('center', **center)
            model.set_param_hint('height', **config.height)
            model.set_param_hint('amplitude', **config.amplitude)
            
            if by == "guess":
                guess_params = model.guess(
                    spec.spec[p-config.peak_window:p+config.peak_window],
                    spec.ws[p-config.peak_window:p+config.peak_window]
                )
            
                params, sub_models, composite_model = _update(model,guess_params, params, sub_models, composite_model)
                
            else:
                default_params = _default_params_from_peaks(m_type, prefix, p_x, p_w, p_h, config)
                params, sub_models, composite_model = _update(model,default_params, params, sub_models, composite_model)

        
        # add lm_models.PolynomialModel() for background
        
        model = lm_models.PolynomialModel(degree=config.poly_n)
        guess_params = model.guess(spec.spec[bg_mask], spec.ws[bg_mask])

        params, sub_models, composite_model = _update(model, guess_params, params, sub_models, composite_model)
        
        # add vogit background? for liquid phase disentanglement -> big spread that fit nicely by vogit!!!
        if config.add_vogit_bg:
            model = lm_models.VoigtModel(prefix="Lb0_")
            
            model.set_param_hint('center', **config.center)
            model.set_param_hint('height', **config.height)
            model.set_param_hint('amplitude', **config.amplitude)
            
#             guess_params = model.guess(spec.spec[bg_mask], spec.ws[bg_mask])
            guess_params = model.guess(spec.spec, spec.ws)
#             guess_params['Lb0_amplitude'].value = config.amplitude['max'] * config.vogit_bg_amp_ratio
    
            params, sub_models, composite_model = _update(model, guess_params, params, sub_models, composite_model)
        
        return cls(composite_model, sub_models, params)    
        
    def modify_params(self, model_idx, **kargs):
        model = self.sub_models[model_idx]
        for param, options in kargs.items():
            model.set_param_hint(param, **options)
            
        params = model.make_params()
        self.params.update(params)
        
        return self
    
    def fit(self, spec, **kargs):
        output = self.model.fit(
            data = spec.spec,
            params= self.params,
            x = spec.ws,
            **kargs
        )
        return output
    
    def plot_component(self, spec, xlabel=None, ylabel=None, ax=None, **kargs):
        fig, ax = _create_figure(ax=ax, **kargs)
        ax.scatter(spec.ws, spec.spec, s=4)
        components = self.output_fit.eval_components(x=spec.ws)

        for k, v in components.items():
            ax.plot(spec.ws, v, label=k)
        if xlabel:
            ax.set_xlabel()
        if ylabel:
            ax.set_ylabel()
        ax.legend()

        return fig, ax

    def plot_fit(self, spec, **kargs):
        fig, gridspec = self.output_fit.plot(data_kws={'markersize': 1}, **kargs)
        fig.axes[0].title.set_text("Fit and Residual")
        return fig, gridspec

class Timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def fit_single(
    spec, config,
    smooth=4,
    bg_thres=0.05,
    height=0.01,
    threshold=0,
    prominence=0.01,
    auto_amplitude=False,
    timeout=3,
    fit_method="leastsq"
):
        
    finished= False

    # find good initialization
    preprocessed = spec.remove_background(inplace=False).normalization(False).smooth(smooth, False)
    peaks, peaksinfo = preprocessed.fit_spectrum_peaks(height=height, threshold=threshold, prominence=prominence)
    
    spec.peaks = peaks

    # preprocessed.plot_spectrum(peaks=peaks)
    bg_mask = preprocessed.spec <= bg_thres
    
    # set the maximium by considering the spectrum itself
    if auto_amplitude:
        config.amplitude["max"] = maximum_amplitude(spec)

    # fit model
    for method in ["guess", "other"]:
        if finished: break
        try:
            with Timeout(seconds=timeout):
                sm = SpectrumModel.from_peaks(peaks, peaksinfo, spec, config, bg_mask, by=method)
                output= sm.fit(spec, method=fit_method)
                spec.peak_fit_res = output
                finished = True
        except Exception as e:
            print(f"method: {method} error:{e}") # all methods fails

    if not finished:
        spec.peak_fit_res = None
        
    return finished