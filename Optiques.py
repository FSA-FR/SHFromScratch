"""
Optiques.py - Complete Version with Apertures and Diffraction Elements
FR: Module pour la generation et la gestion d'optiques diverses.
    Types supportes: lentilles, miroirs, beamsplitters, fenetres, prismes,
    doublets, diaphragmes, trous de diffraction, reseaux de diffraction.
    Formes d'aperture: circulaire, rectangulaire, carree, elliptique, hexagonale, octogonale.
    Fonctionnalites: WFE, tilt, decentrement, affichage automatique.

EN: Module for generating and managing various optical elements.
    Supported types: lenses, mirrors, beamsplitters, windows, prisms,
    doublets, aperture stops, diffraction holes, diffraction gratings.
    Aperture shapes: circular, rectangular, square, elliptical, hexagonal, octagonal.
    Features: WFE, tilt, decentering, automatic display.
"""

import numpy as np
import logging
import os
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Import dependencies
try:
    from Material_Behaviour import MaterialBehaviour, STANDARD_TEMPERATURE_K, Polarization
    MATERIAL_BEHAVIOUR_AVAILABLE = True
except ImportError:
    MATERIAL_BEHAVIOUR_AVAILABLE = False
    STANDARD_TEMPERATURE_K = 293.15

try:
    from Beam import Beam
    BEAM_AVAILABLE = True
except ImportError:
    BEAM_AVAILABLE = False

try:
    from MathAndPhysicsTools import create_grid, zernike_polynomial
    MATH_TOOLS_AVAILABLE = True
except ImportError:
    MATH_TOOLS_AVAILABLE = False

try:
    from Visualization import plot_intensity, plot_phase, plot_beam_map
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from Propagation import Propagation
    PROPAGATION_AVAILABLE = True
except ImportError:
    PROPAGATION_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Optiques")


# =============================================================================
# ENUMS
# =============================================================================

class OpticType(Enum):
    IDEAL_LENS = "ideal_lens"
    SIMPLE_LENS = "simple_lens"
    DOUBLE_LENS = "double_lens"
    DOUBLET_LENS = "doublet_lens"
    MIRROR = "mirror"
    BEAMSPLITTER = "beamsplitter"
    WINDOW = "window"
    PRISM = "prism"
    ASPHERIC_LENS = "aspheric_lens"
    APERTURE_STOP = "aperture_stop"
    DIFFRACTION_HOLE = "diffraction_hole"
    DIFFRACTION_GRATING = "diffraction_grating"


class LensType(Enum):
    PLAN_CONVEX = "plan_convex"
    BICONVEX = "biconvex"
    BICONCAVE = "biconcave"
    PLAN_CONCAVE = "plan_concave"
    MENISCUS = "meniscus"
    PLAN_PLAN = "plan_plan"


class MirrorType(Enum):
    FLAT = "flat"
    SPHERICAL = "spherical"
    PARABOLIC = "parabolic"
    ELLIPTICAL = "elliptical"


class BeamsplitterType(Enum):
    PLATE = "plate"
    CUBE = "cube"
    POLARIZING = "polarizing"


class GratingType(Enum):
    AMPLITUDE = "amplitude"
    PHASE = "phase"
    BLAZED = "blazed"


class WFESource(Enum):
    SURFACE_ROUGHNESS = "surface_roughness"
    PARALLELISM = "parallelism"
    ZERNIKE = "zernike"
    CUSTOM = "custom"
    FILE = "file"


class ApertureShape(Enum):
    CIRCULAR = "circular"
    RECTANGULAR = "rectangular"
    SQUARE = "square"
    ELLIPTICAL = "elliptical"
    HEXAGONAL = "hexagonal"
    OCTAGONAL = "octagonal"
    TRIANGULAR = "triangular"
    SLIT = "slit"


# =============================================================================
# WAVEFRONT ERROR
# =============================================================================

@dataclass
class WaveFrontError:
    surface_roughness_nm: float = 0.0
    parallelism_arcsec: float = 0.0
    zernike_coefficients: Dict[Tuple[int, int], float] = field(default_factory=dict)
    custom_phase_map: Optional[np.ndarray] = None
    wfe_source: WFESource = WFESource.SURFACE_ROUGHNESS

    def generate_phase_map(self, gx: np.ndarray, gy: np.ndarray, wl: float, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None: np.random.seed(seed)
        pm = np.zeros_like(gx)
        if self.surface_roughness_nm > 0:
            noise = np.random.normal(0, self.surface_roughness_nm, gx.shape)
            if gx.shape[0] > 1 and gx.shape[1] > 1:
                try:
                    from scipy.fft import fft2, ifft2
                    nf = fft2(noise); r, c = nf.shape; cr, cc = r//2, c//2
                    m = np.ones_like(noise); rad = min(r, c)//10
                    y, x = np.ogrid[:r, :c]; m[(x-cc)**2 + (y-cr)**2 <= rad**2] = 0
                    nf = np.real(ifft2(nf * m)); crms = np.std(nf)
                    if crms > 0: nf *= self.surface_roughness_nm / crms
                    pm += nf
                except ImportError: pm += noise
            else: pm += noise
        if self.parallelism_arcsec > 0:
            pr = self.parallelism_arcsec * np.pi / (180 * 3600)
            pm += (gx * np.cos(np.pi/4) + gy * np.sin(np.pi/4)) * pr * wl / 1e3
        if self.zernike_coefficients and MATH_TOOLS_AVAILABLE:
            for (n, m), c in self.zernike_coefficients.items():
                pm += c * zernike_polynomial(n, m, gx, gy)
        if self.custom_phase_map is not None:
            if self.custom_phase_map.shape != gx.shape:
                try:
                    from scipy.interpolate import interp2d
                    f = interp2d(gx[0,:], gy[:,0], self.custom_phase_map, kind='cubic')
                    pm += f(np.linspace(gx.min(), gx.max(), gx.shape[1]),
                           np.linspace(gy.min(), gy.max(), gx.shape[0]))
                except: pass
            else: pm += self.custom_phase_map
        return pm


@dataclass
class OpticSpecifications:
    diameter_mm: float
    thickness_mm: float
    material_name: str = "Fused_Silica"
    surface_roughness_nm: float = 1.0
    parallelism_arcsec: float = 10.0
    clear_aperture_ratio: float = 0.9
    edge_thickness_mm: Optional[float] = None
    aperture_shape: ApertureShape = ApertureShape.CIRCULAR
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None


# =============================================================================
# BASE CLASS
# =============================================================================

@dataclass
class OpticalElement(ABC):
    name: str
    specifications: OpticSpecifications
    position_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tilt_deg: Tuple[float, float] = (0.0, 0.0)
    decentering_mm: Tuple[float, float] = (0.0, 0.0)
    temperature_K: float = STANDARD_TEMPERATURE_K
    wavelength_nm: float = 633.0
    display: bool = False
    display_dir: str = "output"
    material: Optional[any] = None
    wfe: Optional[WaveFrontError] = None

    def __post_init__(self):
        if MATERIAL_BEHAVIOUR_AVAILABLE:
            try: self.material = MaterialBehaviour(self.specifications.material_name)
            except: self.material = None
        else: self.material = None
        self.wfe = WaveFrontError(
            surface_roughness_nm=self.specifications.surface_roughness_nm,
            parallelism_arcsec=self.specifications.parallelism_arcsec
        )
        if self.display: os.makedirs(self.display_dir, exist_ok=True)

    @abstractmethod
    def get_phase_map(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray: pass

    def get_aperture_mask(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        s = self.specifications.aperture_shape
        cd = self.specifications.diameter_mm * self.specifications.clear_aperture_ratio
        xc, yc = self.decentering_mm
        x, y = gx - xc, gy - yc
        if s == ApertureShape.CIRCULAR: return (np.sqrt(x**2 + y**2) <= cd/2).astype(float)
        elif s == ApertureShape.RECTANGULAR:
            w = self.specifications.width_mm or cd
            h = self.specifications.height_mm or cd
            return ((np.abs(x) <= w/2) & (np.abs(y) <= h/2)).astype(float)
        elif s == ApertureShape.SQUARE:
            sz = self.specifications.width_mm or cd
            return ((np.abs(x) <= sz/2) & (np.abs(y) <= sz/2)).astype(float)
        elif s == ApertureShape.ELLIPTICAL:
            a = (self.specifications.width_mm or cd)/2
            b = (self.specifications.height_mm or cd)/2
            return ((x**2/a**2) + (y**2/b**2) <= 1.0).astype(float)
        elif s == ApertureShape.HEXAGONAL:
            sz = cd/2
            return (np.abs(x) + np.abs(y)/np.sqrt(3) <= sz*2/np.sqrt(3)).astype(float)
        elif s == ApertureShape.OCTAGONAL:
            sz = cd/2
            return ((np.abs(x) + np.abs(y) <= sz*np.sqrt(2)*(1+np.sqrt(2))) &
                   (np.abs(x) <= sz*(1+np.sqrt(2))) &
                   (np.abs(y) <= sz*(1+np.sqrt(2)))).astype(float)
        elif s == ApertureShape.TRIANGULAR:
            sz = cd/2
            return ((y >= -sz) & (np.abs(x) <= sz*(1 - y/sz)) & (y <= sz)).astype(float)
        elif s == ApertureShape.SLIT:
            w = self.specifications.width_mm or cd*0.1
            h = self.specifications.height_mm or cd
            return ((np.abs(x) <= w/2) & (np.abs(y) <= h/2)).astype(float)
        return (np.sqrt(x**2 + y**2) <= cd/2).astype(float)

    def get_transmission_map(self, gx: np.ndarray, gy: np.ndarray, *a, **kw) -> np.ndarray:
        if self.material is None: return self.get_aperture_mask(gx, gy)
        try:
            t = self.specifications.thickness_mm * 1e-3
            a = self.material._get_absorption_coefficient(self.wavelength_nm)
            return np.exp(-a * t) * self.get_aperture_mask(gx, gy)
        except: return self.get_aperture_mask(gx, gy)

    def get_reflection_map(self, gx: np.ndarray, gy: np.ndarray, *a, **kw) -> np.ndarray:
        if self.material is None: return np.zeros_like(gx)
        try: return self.material.get_reflectance(self.wavelength_nm, *a, **kw) * self.get_aperture_mask(gx, gy)
        except: return np.zeros_like(gx)

    def get_full_phase_map(self, gx: np.ndarray, gy: np.ndarray, iw: bool = True, it: bool = True, s: Optional[int] = None) -> np.ndarray:
        pm = self.get_phase_map(gx, gy)
        if iw and self.wfe: pm += self.wfe.generate_phase_map(gx, gy, self.wavelength_nm, s)
        if it and (self.tilt_deg[0] or self.tilt_deg[1]):
            tx, ty = np.deg2rad(self.tilt_deg[0]), np.deg2rad(self.tilt_deg[1])
            pm += (gx * tx + gy * ty) * self.wavelength_nm
        return pm * self.get_aperture_mask(gx, gy)

    def apply_to_beam(self, beam: any) -> any:
        if not BEAM_AVAILABLE: raise ImportError("Beam module required")
        if MATH_TOOLS_AVAILABLE: gx, gy = create_grid(beam.diameter_mm, beam.num_points)
        else:
            x = y = np.linspace(-beam.diameter_mm/2, beam.diameter_mm/2, beam.num_points)
            gx, gy = np.meshgrid(x, y)
        if self.display and VISUALIZATION_AVAILABLE:
            os.makedirs(self.display_dir, exist_ok=True)
            if hasattr(beam, 'intensity') and beam.intensity is not None:
                try: plot_intensity(beam.intensity, beam.diameter_mm, title=f"Before {self.name}",
                                     save_path=os.path.join(self.display_dir, f"before_{self.name}_intensity.png"))
                except: pass
            if hasattr(beam, 'phase') and beam.phase is not None:
                try: plot_phase(beam.phase, beam.diameter_mm, title=f"Before {self.name}",
                               save_path=os.path.join(self.display_dir, f"before_{self.name}_phase.png"))
                except: pass
        pm = self.get_full_phase_map(gx, gy)
        nb = Beam(beam.wavelength_nm, beam.diameter_mm, beam.energy, beam.num_points, beam.coherence)
        if beam.electric_field is not None:
            amp = np.abs(beam.electric_field)
            ip = np.angle(beam.electric_field)
            nb.electric_field = amp * np.exp(1j * (ip + pm * 2 * np.pi / self.wavelength_nm))
            try:
                nb.intensity = nb.compute_intensity_from_electric_field(nb.electric_field)
                nb.phase = nb.extract_phase_from_electric_field(nb.electric_field)
            except: pass
        if self.display and VISUALIZATION_AVAILABLE:
            if hasattr(nb, 'intensity') and nb.intensity is not None:
                try: plot_intensity(nb.intensity, nb.diameter_mm, title=f"After {self.name}",
                                     save_path=os.path.join(self.display_dir, f"after_{self.name}_intensity.png"))
                except: pass
            if hasattr(nb, 'phase') and nb.phase is not None:
                try: plot_phase(nb.phase, nb.diameter_mm, title=f"After {self.name}",
                               save_path=os.path.join(self.display_dir, f"after_{self.name}_phase.png"))
                except: pass
        return nb

    def get_optical_path_difference(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        if self.material is None: return np.zeros_like(gx)
        try: n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        except: n = 1.5
        return (n - 1) * self._get_effective_thickness(gx, gy) * 1e6

    @abstractmethod
    def _get_effective_thickness(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray: pass


# =============================================================================
# SPECIFIC CLASSES
# =============================================================================

class IdealLens(OpticalElement):
    def __init__(self, name: str, f: float, d: float, mat: str = "ideal", s: Optional[OpticSpecifications] = None, **k):
        if s is None: s = OpticSpecifications(d, 0.0, mat)
        self.focal_length_mm = f
        super().__init__(name=name, specifications=s, **k)
    def get_phase_map(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        fm = self.focal_length_mm * 1e-3
        xm, ym = gx * 1e-3, gy * 1e-3
        lm = self.wavelength_nm * 1e-9
        return - (2 * np.pi / lm) * (xm**2 + ym**2) / (2 * fm) * lm / (2 * np.pi) * 1e9
    def _get_effective_thickness(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        return np.full_like(gx, 0.0)


class SimpleLens(OpticalElement):
    def __init__(self, name: str, R: float, d: float, t: float, lt: LensType = LensType.PLAN_CONVEX,
                 mat: str = "BK7", cfp: str = "front", s: Optional[OpticSpecifications] = None, **k):
        if s is None: s = OpticSpecifications(d, t, mat)
        self.R_mm, self.lens_type, self.cfp = R, lt, cfp
        super().__init__(name=name, specifications=s, **k)
    def get_phase_map(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        if self.material is None: return np.zeros_like(gx)
        return (self.material.get_refractive_index(self.wavelength_nm, self.temperature_K) - 1) * self._get_effective_thickness(gx, gy) * 1e6
    def _get_effective_thickness(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        R, d = self.R_mm, self.specifications.thickness_mm
        r = np.sqrt(gx**2 + gy**2)
        sign = -1 if (self.lens_type == LensType.PLAN_CONVEX and self.cfp == "front") or (self.lens_type == LensType.PLAN_CONCAVE and self.cfp == "back") else 1
        return np.maximum(d + sign * r**2 / (2 * abs(R)), 0.0)


class DoubleLens(OpticalElement):
    def __init__(self, name: str, R1: float, R2: float, d: float, t: float, lt: LensType = LensType.BICONVEX,
                 mat: str = "BK7", s: Optional[OpticSpecifications] = None, **k):
        if s is None: s = OpticSpecifications(d, t, mat)
        self.R1, self.R2, self.lens_type = R1, R2, lt
        super().__init__(name=name, specifications=s, **k)
    def get_phase_map(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        if self.material is None: return np.zeros_like(gx)
        return (self.material.get_refractive_index(self.wavelength_nm, self.temperature_K) - 1) * self._get_effective_thickness(gx, gy) * 1e6
    def _get_effective_thickness(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        R1, R2, d = self.R1, self.R2, self.specifications.thickness_mm
        r = np.sqrt(gx**2 + gy**2)
        if self.lens_type == LensType.BICONVEX: t = d - r**2/(2*R1) - r**2/(2*R2)
        elif self.lens_type == LensType.BICONCAVE: t = d + r**2/(2*abs(R1)) + r**2/(2*abs(R2))
        elif self.lens_type == LensType.MENISCUS: t = d - r**2/(2*R1) + r**2/(2*R2)
        else: t = np.full_like(gx, d)
        return np.maximum(t, 0.0)


class DoubletLens(OpticalElement):
    def __init__(self, name: str, D: float, R1: float, R2: float, t1: float, m1: str = "BK7",
                 R3: float, R4: float, t2: float, m2: str = "SF5", s: Optional[OpticSpecifications] = None, **k):
        if s is None: s = OpticSpecifications(D, t1+t2, f"{m1}_{m2}")
        super().__init__(name=name, specifications=s, **k)
        self.R1, self.R2, self.R3, self.R4 = R1, R2, R3, R4
        self.d1, self.d2 = t1, t2
        self.material1 = MaterialBehaviour(m1) if MATERIAL_BEHAVIOUR_AVAILABLE else None
        self.material2 = MaterialBehaviour(m2) if MATERIAL_BEHAVIOUR_AVAILABLE else None
    def get_phase_map(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        if self.material1 is None or self.material2 is None: return np.zeros_like(gx)
        n1 = self.material1.get_refractive_index(self.wavelength_nm, self.temperature_K)
        n2 = self.material2.get_refractive_index(self.wavelength_nm, self.temperature_K)
        return (n1-1)*self._get_et1(gx, gy)*1e6 + (n2-1)*self._get_et2(gx, gy)*1e6
    def _get_effective_thickness(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        return self._get_et1(gx, gy) + self._get_et2(gx, gy)
    def _get_et1(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        R1, R2, d1 = self.R1, self.R2, self.d1
        r = np.sqrt(gx**2 + gy**2)
        if R1 > 0 and R2 > 0: t = d1 - r**2/(2*R1) - r**2/(2*R2)
        elif R1 > 0 and R2 < 0:
            if abs(R2) > 1e6: t = d1 - r**2/(2*R1)
            else: t = d1 - r**2/(2*R1) + r**2/(2*abs(R2))
        else: t = np.full_like(gx, d1)
        return np.maximum(t, 0.0)
    def _get_et2(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        R3, R4, d2 = self.R3, self.R4, self.d2
        r = np.sqrt(gx**2 + gy**2)
        if R3 < 0 and R4 < 0: t = d2 + r**2/(2*abs(R3)) + r**2/(2*abs(R4))
        elif R3 < 0 and abs(R4) > 1e6: t = d2 + r**2/(2*abs(R3))
        else: t = np.full_like(gx, d2)
        return np.maximum(t, 0.0)
    @staticmethod
    def create_achromatic_doublet(f: float, D: float, m1: str = "BK7", m2: str = "SF5", wl: float = 633.0, **k) -> 'DoubletLens':
        if not MATERIAL_BEHAVIOUR_AVAILABLE: raise ImportError("Material_Behaviour required")
        n1, n2 = MaterialBehaviour(m1).get_refractive_index(wl), MaterialBehaviour(m2).get_refractive_index(wl)
        V1, V2 = 64.0, 32.0
        P = 1/(f*1e-3); P1, P2 = P*V1/(V1-V2), P*V2/(V2-V1)
        R1, R2, R3, R4 = 2*(n1-1)/P1, -R1, -R2, 2*(n2-1)/P2
        return DoubletLens(f"Achromatic_f{f}mm", D, R1, R2, 3.0, m1, R3, R4, 2.0, m2, wavelength_nm=wl, **k)


class Mirror(OpticalElement):
    def __init__(self, name: str, D: float, mt: MirrorType = MirrorType.FLAT, R: Optional[float] = None,
                 f: Optional[float] = None, mat: str = "Aluminum", refl: float = 0.95,
                 s: Optional[OpticSpecifications] = None, **k):
        if s is None: s = OpticSpecifications(D, 10.0, mat)
        super().__init__(name=name, specifications=s, **k)
        self.mirror_type, self.R_mm, self.f_mm = mt, R, f
        self.coating_reflectivity = refl
        if f is not None and R is None:
            self.R_mm = 2*f if mt in [MirrorType.SPHERICAL, MirrorType.PARABOLIC] else None
    def get_phase_map(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        if self.mirror_type == MirrorType.FLAT: return np.zeros_like(gx)
        elif self.mirror_type == MirrorType.SPHERICAL:
            if self.R_mm is None: return np.zeros_like(gx)
            return -2*(gx**2 + gy**2)/self.R_mm * 1e6
        elif self.mirror_type == MirrorType.PARABOLIC:
            if self.f_mm is None: return np.zeros_like(gx)
            return -(gx**2 + gy**2)/self.f_mm * 1e6
        return np.zeros_like(gx)
    def get_reflection_map(self, gx: np.ndarray, gy: np.ndarray, *a, **kw) -> np.ndarray:
        return self.coating_reflectivity * self.get_aperture_mask(gx, gy)
    def _get_effective_thickness(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        return np.full_like(gx, self.specifications.thickness_mm)


class Beamsplitter(OpticalElement):
    def __init__(self, name: str, D: float, bt: BeamsplitterType = BeamsplitterType.PLATE,
                 trans: float = 0.5, refl: float = 0.5, mat: str = "BK7", pa: Optional[str] = None,
                 s: Optional[OpticSpecifications] = None, **k):
        if s is None: s = OpticSpecifications(D, 1.0, mat)
        super().__init__(name=name, specifications=s, **k)
        self.beamsplitter_type = bt
        t = trans + refl
        self.transmission_ratio = trans/t if t > 0 else 0.5
        self.reflection_ratio = refl/t if t > 0 else 0.5
        self.polarization_axis = pa
    def get_phase_map(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        if self.material is None: return np.zeros_like(gx)
        return (self.material.get_refractive_index(self.wavelength_nm, self.temperature_K) - 1) * self._get_effective_thickness(gx, gy) * 1e6
    def get_transmission_map(self, gx: np.ndarray, gy: np.ndarray, *a, **kw) -> np.ndarray:
        m = self.get_aperture_mask(gx, gy)
        if self.beamsplitter_type == BeamsplitterType.POLARIZING and self.polarization_axis:
            return m * (0.5 if kw.get('polarization') is None else 1.0)
        return self.transmission_ratio * m
    def get_reflection_map(self, gx: np.ndarray, gy: np.ndarray, *a, **kw) -> np.ndarray:
        m = self.get_aperture_mask(gx, gy)
        if self.beamsplitter_type == BeamsplitterType.POLARIZING and self.polarization_axis:
            return m * (0.5 if kw.get('polarization') is None else 1.0)
        return self.reflection_ratio * m
    def _get_effective_thickness(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        return np.full_like(gx, self.specifications.thickness_mm)


class Window(OpticalElement):
    def __init__(self, name: str, D: float, t: float, mat: str = "Fused_Silica",
                 s: Optional[OpticSpecifications] = None, **k):
        if s is None: s = OpticSpecifications(D, t, mat)
        super().__init__(name=name, specifications=s, **k)
    def get_phase_map(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        if self.material is None: return np.zeros_like(gx)
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        phase = (n-1)*self.specifications.thickness_mm*1e6
        if self.tilt_deg[0] or self.tilt_deg[1]:
            tx, ty = np.deg2rad(self.tilt_deg[0]), np.deg2rad(self.tilt_deg[1])
            phase += (n-1)*(gx*tx + gy*ty)*1e3
        return phase
    def _get_effective_thickness(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        return np.full_like(gx, self.specifications.thickness_mm)


class Prism(OpticalElement):
    def __init__(self, name: str, apex: float, base: float, height: float, mat: str = "BK7",
                 orient: float = 0.0, s: Optional[OpticSpecifications] = None, **k):
        D = np.sqrt(base**2 + height**2)
        if s is None: s = OpticSpecifications(D, height, mat, aperture_shape=ApertureShape.TRIANGULAR, width_mm=base, height_mm=height)
        super().__init__(name=name, specifications=s, **k)
        self.apex_angle_deg, self.apex_angle_rad = apex, np.deg2rad(apex)
        self.base_length_mm, self.height_mm = base, height
        self.orientation_deg, self.orientation_rad = orient, np.deg2rad(orient)
    def get_phase_map(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        if self.material is None: return np.zeros_like(gx)
        return (self.material.get_refractive_index(self.wavelength_nm, self.temperature_K) - 1) * self._get_effective_thickness(gx, gy) * 1e6
    def get_deviation_angle(self, wl: Optional[float] = None) -> float:
        if wl is None: wl = self.wavelength_nm
        if self.material is None: return 0.0
        return np.rad2deg((self.material.get_refractive_index(wl, self.temperature_K) - 1) * self.apex_angle_rad)
    def _get_effective_thickness(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        a = self.orientation_rad
        yp = gx*np.sin(a) + gy*np.cos(a)
        return np.maximum(self.height_mm/2 - yp*np.tan(self.apex_angle_rad/2), 0.0)


class AsphericLens(OpticalElement):
    def __init__(self, name: str, R: float, D: float, t: float, k: float = 0.0,
                 coeffs: Optional[Dict[int, float]] = None, mat: str = "BK7",
                 s: Optional[OpticSpecifications] = None, **kw):
        if s is None: s = OpticSpecifications(D, t, mat)
        super().__init__(name=name, specifications=s, **kw)
        self.R_mm, self.k, self.coeffs = R, k, coeffs or {}
    def get_phase_map(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        if self.material is None: return np.zeros_like(gx)
        return (self.material.get_refractive_index(self.wavelength_nm, self.temperature_K) - 1) * self._get_effective_thickness(gx, gy) * 1e6
    def _get_effective_thickness(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        r = np.sqrt(gx**2 + gy**2)
        R = self.R_mm
        with np.errstate(divide='ignore', invalid='ignore'):
            sag = (r**2/R)/(1+np.sqrt(1-(1+self.k)*(r**2/R**2)))
            for i, a in self.coeffs.items(): sag += a*(r**i)
            sag = np.nan_to_num(sag, 0)
        return np.maximum(self.specifications.thickness_mm - sag, 0.0)


class ApertureStop(OpticalElement):
    """Diaphragme - bloque la lumiere en dehors de l'ouverture."""
    def __init__(self, name: str, D: float, mat: str = "opaque",
                 s: Optional[OpticSpecifications] = None, **k):
        if s is None: s = OpticSpecifications(D, 0.1, mat)
        super().__init__(name=name, specifications=s, **k)
    def get_phase_map(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        return np.zeros_like(gx)
    def get_transmission_map(self, gx: np.ndarray, gy: np.ndarray, *a, **kw) -> np.ndarray:
        return self.get_aperture_mask(gx, gy)
    def get_reflection_map(self, gx: np.ndarray, gy: np.ndarray, *a, **kw) -> np.ndarray:
        return np.zeros_like(gx)
    def _get_effective_thickness(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        return np.full_like(gx, self.specifications.thickness_mm)


class DiffractionHole(OpticalElement):
    """Trou de diffraction - pour figures de diffraction."""
    def __init__(self, name: str, D: float, mat: str = "air",
                 s: Optional[OpticSpecifications] = None, **k):
        if s is None: s = OpticSpecifications(D, 0.0, mat)
        super().__init__(name=name, specifications=s, **k)
    def get_phase_map(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        return np.zeros_like(gx)
    def get_transmission_map(self, gx: np.ndarray, gy: np.ndarray, *a, **kw) -> np.ndarray:
        return self.get_aperture_mask(gx, gy)
    def get_diffraction_pattern(self, dist: float, gs: float = 10.0, np_pts: int = 256) -> np.ndarray:
        if not PROPAGATION_AVAILABLE:
            logger.warning("Propagation module not available"); return np.zeros((np_pts, np_pts))
        if MATH_TOOLS_AVAILABLE: gx, gy = create_grid(gs, np_pts)
        else:
            x = y = np.linspace(-gs/2, gs/2, np_pts); gx, gy = np.meshgrid(x, y)
        field = self.get_aperture_mask(gx, gy).astype(complex)
        try:
            prop = Propagation(self.wavelength_nm, dist, gs, gs, np_pts)
            return np.abs(prop.propagate(field))**2
        except Exception as e:
            logger.warning(f"Diffraction failed: {e}"); return np.zeros_like(gx)
    def _get_effective_thickness(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        return np.zeros_like(gx)


class DiffractionGrating(OpticalElement):
    """Reseau de diffraction - devie la lumiere selon d*sin(theta) = m*lambda."""
    def __init__(self, name: str, D: float, lines: float = 100.0, orient: float = 0.0,
                 gtype: GratingType = GratingType.AMPLITUDE, blaze: float = 0.0,
                 mat: str = "BK7", s: Optional[OpticSpecifications] = None, **k):
        if s is None: s = OpticSpecifications(D, 0.5, mat)
        super().__init__(name=name, specifications=s, **k)
        self.lines_per_mm, self.orientation_deg = lines, orient
        self.orientation_rad = np.deg2rad(orient)
        self.grating_type, self.blaze_angle_deg = gtype, blaze
        self.blaze_angle_rad = np.deg2rad(blaze)
    def get_phase_map(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        if self.material is None: return np.zeros_like(gx)
        lm = self.wavelength_nm * 1e-9; k = 2*np.pi/lm
        xr = gx*np.cos(self.orientation_rad) - gy*np.sin(self.orientation_rad)
        d = 1.0/self.lines_per_mm
        if self.grating_type == GratingType.PHASE:
            try: n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
            except: n = 1.5
            return (n-1)*self.specifications.thickness_mm*1e6*np.sin(2*np.pi*xr/d)
        elif self.grating_type == GratingType.BLAZED:
            return (k*xr*np.sin(self.blaze_angle_rad)/d)*lm/(2*np.pi)*1e9
        return np.zeros_like(gx)
    def get_transmission_map(self, gx: np.ndarray, gy: np.ndarray, *a, **kw) -> np.ndarray:
        m = self.get_aperture_mask(gx, gy)
        if self.grating_type == GratingType.AMPLITUDE:
            xr = gx*np.cos(self.orientation_rad) - gy*np.sin(self.orientation_rad)
            d = 1.0/self.lines_per_mm
            return ((xr % d) < (d/2)).astype(float) * m
        return m
    def get_diffraction_orders(self, max_order: int = 3) -> Dict[int, Optional[float]]:
        d = 1.0/self.lines_per_mm; lm = self.wavelength_nm * 1e-9
        return {m: np.rad2deg(np.arcsin(m*lm/(d*1e-3))) if abs(m*lm/(d*1e-3)) <= 1.0 else None
                for m in range(-max_order, max_order+1)}
    def _get_effective_thickness(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        return np.full_like(gx, self.specifications.thickness_mm)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_optic(otype: OpticType, name: str = "Optic", **kw) -> OpticalElement:
    classes = {
        OpticType.IDEAL_LENS: IdealLens, OpticType.SIMPLE_LENS: SimpleLens,
        OpticType.DOUBLE_LENS: DoubleLens, OpticType.DOUBLET_LENS: DoubletLens,
        OpticType.MIRROR: Mirror, OpticType.BEAMSPLITTER: Beamsplitter,
        OpticType.WINDOW: Window, OpticType.PRISM: Prism, OpticType.ASPHERIC_LENS: AsphericLens,
        OpticType.APERTURE_STOP: ApertureStop, OpticType.DIFFRACTION_HOLE: DiffractionHole,
        OpticType.DIFFRACTION_GRATING: DiffractionGrating
    }
    if otype not in classes: raise ValueError(f"Unknown optic type: {otype}")
    return classes[otype](name=name, **kw)


def create_lens_from_preset(preset: str, D: float, f: float, mat: str = "BK7", **kw) -> OpticalElement:
    presets = {
        "ideal": {"type": OpticType.IDEAL_LENS},
        "plan_convex": {"type": OpticType.SIMPLE_LENS, "R": lambda: (1.5-1)*f, "lt": LensType.PLAN_CONVEX, "cfp": "front"},
        "biconvex": {"type": OpticType.DOUBLE_LENS, "R1": lambda: 2*(1.5-1)*f, "R2": lambda: -2*(1.5-1)*f, "lt": LensType.BICONVEX},
        "achromatic_doublet": {"factory": DoubletLens.create_achromatic_doublet},
        "window": {"type": OpticType.WINDOW, "t": 2.0},
        "prism_45": {"type": OpticType.PRISM, "apex": 45.0, "base": 10.0, "height": 10.0},
        "mirror_flat": {"type": OpticType.MIRROR, "mt": MirrorType.FLAT},
        "beamsplitter": {"type": OpticType.BEAMSPLITTER, "bt": BeamsplitterType.PLATE},
        "aperture_stop": {"type": OpticType.APERTURE_STOP},
        "diffraction_hole": {"type": OpticType.DIFFRACTION_HOLE},
        "diffraction_grating": {"type": OpticType.DIFFRACTION_GRATING, "lines": 100.0},
    }
    if preset not in presets: raise ValueError(f"Unknown preset: {preset}")
    p = presets[preset]
    if "factory" in p: return p["factory"](f_mm=f, diameter_mm=D, material1_name=mat, material2_name="SF5" if preset == "achromatic_doublet" else mat, **kw)
    otype = p["type"]
    opts = {k: v() if callable(v) else v for k, v in p.items() if k not in ["type", "factory"]}
    kw = {"name": f"{preset}_{D}mm", "diameter_mm": D, **opts, **kw}
    if otype == OpticType.IDEAL_LENS: kw["f"] = f
    return create_optic(otype, **kw)


# =============================================================================
# OPTICAL SYSTEM
# =============================================================================

class OpticalSystem:
    def __init__(self, display: bool = False, display_dir: str = "output"):
        self.elements: List[OpticalElement] = []
        self.display = display
        self.display_dir = display_dir
        if display: os.makedirs(display_dir, exist_ok=True)

    def add_element(self, e: OpticalElement): self.elements.append(e)

    def add_lens(self, name: str, f: float, D: float, mat: str = "BK7", z: float = 0.0, **kw) -> OpticalElement:
        l = IdealLens(name, f, D, mat, position_mm=(0,0,z), **kw)
        self.add_element(l); return l

    def add_mirror(self, name: str, D: float, mt: MirrorType = MirrorType.FLAT, R: Optional[float] = None,
                   f: Optional[float] = None, z: float = 0.0, tilt: Tuple[float, float] = (0,0), **kw) -> OpticalElement:
        m = Mirror(name, D, mt, R, f, position_mm=(0,0,z), tilt_deg=tilt, **kw)
        self.add_element(m); return m

    def add_prism(self, name: str, apex: float, base: float, height: float, mat: str = "BK7",
                  z: float = 0.0, orient: float = 0.0, **kw) -> OpticalElement:
        p = Prism(name, apex, base, height, mat, orient, position_mm=(0,0,z), **kw)
        self.add_element(p); return p

    def add_aperture_stop(self, name: str, D: float, z: float = 0.0,
                          shape: ApertureShape = ApertureShape.CIRCULAR,
                          w: Optional[float] = None, h: Optional[float] = None, **kw) -> OpticalElement:
        s = OpticSpecifications(D, 0.1, aperture_shape=shape, width_mm=w, height_mm=h)
        a = ApertureStop(name, D, specifications=s, position_mm=(0,0,z), **kw)
        self.add_element(a); return a

    def add_diffraction_hole(self, name: str, D: float, z: float = 0.0,
                              shape: ApertureShape = ApertureShape.CIRCULAR,
                              w: Optional[float] = None, h: Optional[float] = None, **kw) -> OpticalElement:
        s = OpticSpecifications(D, 0.0, aperture_shape=shape, width_mm=w, height_mm=h)
        h = DiffractionHole(name, D, specifications=s, position_mm=(0,0,z), **kw)
        self.add_element(h); return h

    def add_diffraction_grating(self, name: str, D: float, lines: float = 100.0, z: float = 0.0,
                                 orient: float = 0.0, gtype: GratingType = GratingType.AMPLITUDE, **kw) -> OpticalElement:
        g = DiffractionGrating(name, D, lines, orient, gtype, position_mm=(0,0,z), **kw)
        self.add_element(g); return g

    def sort_elements_by_position(self): self.elements.sort(key=lambda x: x.position_mm[2])

    def propagate_beam(self, beam: any, initial_z: float = 0.0, use_prop: bool = True) -> any:
        if not BEAM_AVAILABLE: raise ImportError("Beam module required")
        self.sort_elements_by_position()
        cz = initial_z; cb = beam
        if self.display and VISUALIZATION_AVAILABLE:
            os.makedirs(self.display_dir, exist_ok=True)
            if hasattr(cb, 'intensity') and cb.intensity is not None:
                try: plot_intensity(cb.intensity, cb.diameter_mm, title="Initial Beam",
                                     save_path=os.path.join(self.display_dir, "initial_beam.png"))
                except: pass
        for el in self.elements:
            ez = el.position_mm[2]; dist = ez - cz
            if dist > 0 and use_prop and PROPAGATION_AVAILABLE:
                try:
                    prop = Propagation(beam.wavelength_nm, dist, cb.diameter_mm, cb.diameter_mm, cb.num_points)
                    pf = prop.propagate(cb.electric_field)
                    cb.electric_field = pf
                    try:
                        cb.intensity = cb.compute_intensity_from_electric_field(pf)
                        cb.phase = cb.extract_phase_from_electric_field(pf)
                    except: pass
                except Exception as e: logger.warning(f"Propagation failed: {e}")
                if self.display and VISUALIZATION_AVAILABLE:
                    if hasattr(cb, 'intensity') and cb.intensity is not None:
                        try: plot_intensity(cb.intensity, cb.diameter_mm,
                                              title=f"After Prop to {el.name}",
                                              save_path=os.path.join(self.display_dir, f"after_prop_{el.name}.png"))
                        except: pass
            cb = el.apply_to_beam(cb); cz = ez
        if self.display and VISUALIZATION_AVAILABLE:
            if hasattr(cb, 'intensity') and cb.intensity is not None:
                try: plot_intensity(cb.intensity, cb.diameter_mm, title="Final Beam",
                                     save_path=os.path.join(self.display_dir, "final_beam.png"))
                except: pass
        return cb

    def get_total_phase_map(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        tp = np.zeros_like(gx)
        for el in self.elements: tp += el.get_full_phase_map(gx, gy)
        return tp


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestOptiques:
    def test_ideal_lens(self):
        l = IdealLens("Test", 100, 10)
        gx, gy = (np.linspace(-5,5,256),)*2
        if MATH_TOOLS_AVAILABLE: gx, gy = create_grid(10, 256)
        else: gx, gy = np.meshgrid(*gx)
        p = l.get_phase_map(gx, gy)
        assert np.all(p <= 0) and abs(p[128,128]) < 1e-6

    def test_doublet(self):
        d = DoubletLens("Test", 10, 100, -50, 3, "BK7", 50, -200, 2, "SF5")
        gx, gy = (np.linspace(-5,5,256),)*2
        if MATH_TOOLS_AVAILABLE: gx, gy = create_grid(10, 256)
        else: gx, gy = np.meshgrid(*gx)
        p = d.get_phase_map(gx, gy)
        assert p.shape == gx.shape and not np.all(p == 0)

    def test_aperture_shapes(self):
        for s in list(ApertureShape):
            specs = OpticSpecifications(10, 2, aperture_shape=s, width_mm=8, height_mm=6)
            o = Window("T", 10, 2, specifications=specs)
            gx, gy = (np.linspace(-10,10,256),)*2
            if MATH_TOOLS_AVAILABLE: gx, gy = create_grid(20, 256)
            else: gx, gy = np.meshgrid(*gx)
            m = o.get_aperture_mask(gx, gy)
            assert m.shape == gx.shape and np.any(m > 0)


if __name__ == "__main__":
    import unittest
    unittest.main()
