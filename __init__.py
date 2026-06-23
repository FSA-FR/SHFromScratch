"""
SHFromScratch - Package pour la simulation de systèmes Shack-Hartmann

FR: Package Python pour la simulation complète de systèmes Shack-Hartmann.
    
    Structure du package :
    - MathAndPhysicsTools.py : Fonctions outils mathématiques et physiques
    - Beam.py : Génération et propagation de faisceaux
    - Propagation.py : Propagation à travers des éléments optiques
    - Optiques.py : Éléments optiques (lentilles, diaphragmes, etc.)
    - Microstructure.py : Matrices de microlentilles
    - Camera.py : Capteurs optiques (caméras)
    - Shack_Hartmann.py : Capteur Shack-Hartmann
    - Southwell.py : Reconstruction modale (algorithme de Southwell)
    - Material_Behaviour.py : Comportement des matériaux (indice, thermique)
    - Visualization.py : Fonctions de visualisation
    - Simulation.py : Simulation complète
    
    Unités par défaut :
    - Longueurs : mm
    - Longueur d'onde : nm
    - Phase : nm (principale), rad (pour les calculs)

EN: Python package for complete Shack-Hartmann system simulation.
    
    Package structure:
    - MathAndPhysicsTools.py: Mathematical and physical utility functions
    - Beam.py: Beam generation and propagation
    - Propagation.py: Propagation through optical elements
    - Optiques.py: Optical elements (lenses, diaphragms, etc.)
    - Microstructure.py: Microlens arrays
    - Camera.py: Optical sensors (cameras)
    - Shack_Hartmann.py: Shack-Hartmann sensor
    - Southwell.py: Modal reconstruction (Southwell algorithm)
    - Material_Behaviour.py: Material behavior (refractive index, thermal)
    - Visualization.py: Visualization functions
    - Simulation.py: Complete simulation
    
    Default units:
    - Lengths: mm
    - Wavelength: nm
    - Phase: nm (main), rad (for calculations)

Author: Fabrice Sanson (FSA-FR)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy (>=1.20.0)
    - matplotlib (>=3.4.0)
    - scipy (>=1.7.0, optional for advanced features)
"""

__version__ = "1.0.0"
__author__ = "Fabrice Sanson"
__email__ = "fsanson@imagine-optic.com"
__license__ = "MIT"


# =============================================================================
# IMPORTS PRINCIPAUX
# =============================================================================

# MathAndPhysicsTools
from MathAndPhysicsTools import (
    # Création de grilles
    create_grid,
    create_polar_grid,
    cartesian_to_polar,
    polar_to_cartesian,
    
    # Gestion des NaN
    handle_nan,
    safe_divide,
    safe_sqrt,
    safe_log,
    safe_exp,
    
    # Statistiques
    compute_pv_rms,
    compute_statistics,
    normalize_array,
    
    # Conversions d'unités
    nm_to_rad,
    rad_to_nm,
    nm_to_lambda,
    lambda_to_nm,
    rad_to_mrad,
    mrad_to_rad,
    mm_to_um,
    um_to_mm,
    
    # Génération de modes
    generate_zernike_polynomial,
    generate_zernike_modes,
    generate_legendre_polynomial,
    generate_hermite_polynomial,
    generate_laguerre_polynomial,
    
    # Constantes
    DEFAULT_WAVELENGTH_NM,
    PI,
    TWO_PI,
    
    # Enums
    ZernikeOrdering,
    NormalizationType
)

# Beam
from Beam import (
    Beam,
    BeamProfile,
    PropagationMethod,
    create_beam
)

# Microstructure
from Microstructure import (
    MicrolensArray,
    MicrolensShape,
    ArrayType,
    create_microlens_array
)

# Camera
from Camera import (
    Camera,
    PerfectCamera,
    CameraType,
    SpectralResponse,
    create_camera,
    create_perfect_camera
)

# Shack_Hartmann
from Shack_Hartmann import (
    ShackHartmann,
    Spot,
    ReconstructionMethod,
    CentroidMethod,
    create_shack_hartmann
)

# Southwell
from Southwell import (
    SouthwellReconstructor,
    reconstruct_wavefront_southwell
)

# Material_Behaviour
from Material_Behaviour import (
    Material,
    MaterialDatabase,
    MaterialCategory,
    RefractiveIndexModel,
    ThermalExpansionModel,
    create_material,
    create_material_database
)

# Optiques
from Optiques import (
    WaveFrontError,
    OpticalElement,
    Lens,
    LensType,
    Diaphragm,
    ApertureShape,
    Hole,
    Grating,
    GratingType,
    OpticType,
    create_lens,
    create_diaphragm,
    create_hole,
    create_grating
)

# Propagation
from Propagation import (
    OpticalSystem,
    PropagationMode,
    propagate_through_lens,
    propagate_through_free_space,
    calculate_focal_spot
)

# Visualization
from Visualization import (
    display_2d_array,
    display_phase,
    display_intensity,
    display_slopes,
    display_error,
    display_airy_spots,
    display_reconstruction_results,
    plot_error_vs_amplitude,
    plot_zonal_vs_modal,
    plot_individual_draws,
    plot_error_histogram,
    plot_comparison_histogram
)

# Simulation
from Simulation import (
    Simulation,
    SimulationResult,
    create_simulation
)


# =============================================================================
# __all__
# =============================================================================

__all__ = [
    # Version
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    
    # Fonctions de création
    'create_beam',
    'create_microlens_array',
    'create_shack_hartmann',
    'create_simulation',
    'create_camera',
    'create_perfect_camera',
    'create_lens',
    'create_diaphragm',
    'create_hole',
    'create_grating',
    'create_material',
    'create_material_database',
    
    # Classes
    'Beam',
    'BeamProfile',
    'PropagationMethod',
    'MicrolensArray',
    'MicrolensShape',
    'ArrayType',
    'Camera',
    'PerfectCamera',
    'CameraType',
    'SpectralResponse',
    'ShackHartmann',
    'Spot',
    'ReconstructionMethod',
    'CentroidMethod',
    'SouthwellReconstructor',
    'Simulation',
    'SimulationResult',
    'WaveFrontError',
    'OpticalElement',
    'Lens',
    'LensType',
    'Diaphragm',
    'ApertureShape',
    'Hole',
    'Grating',
    'GratingType',
    'OpticType',
    'OpticalSystem',
    'PropagationMode',
    'Material',
    'MaterialDatabase',
    'MaterialCategory',
    'RefractiveIndexModel',
    'ThermalExpansionModel',
    
    # Fonctions de visualisation
    'display_2d_array',
    'display_phase',
    'display_intensity',
    'display_slopes',
    'display_error',
    'display_airy_spots',
    'display_reconstruction_results',
    'plot_error_vs_amplitude',
    'plot_zonal_vs_modal',
    'plot_individual_draws',
    'plot_error_histogram',
    'plot_comparison_histogram',
    
    # Fonctions outils
    'create_grid',
    'create_polar_grid',
    'cartesian_to_polar',
    'polar_to_cartesian',
    'handle_nan',
    'safe_divide',
    'safe_sqrt',
    'safe_log',
    'safe_exp',
    'compute_pv_rms',
    'compute_statistics',
    'normalize_array',
    'nm_to_rad',
    'rad_to_nm',
    'nm_to_lambda',
    'lambda_to_nm',
    'rad_to_mrad',
    'mrad_to_rad',
    'mm_to_um',
    'um_to_mm',
    'generate_zernike_polynomial',
    'generate_zernike_modes',
    'generate_legendre_polynomial',
    'generate_hermite_polynomial',
    'generate_laguerre_polynomial',
    
    # Constantes
    'DEFAULT_WAVELENGTH_NM',
    'PI',
    'TWO_PI',
    'ZernikeOrdering',
    'NormalizationType',
    
    # Fonctions de Propagation
    'propagate_through_lens',
    'propagate_through_free_space',
    'calculate_focal_spot',
    
    # Fonction de reconstruction
    'reconstruct_wavefront_southwell'
]
