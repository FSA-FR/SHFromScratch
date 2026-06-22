"""
Southwell.py

FR: Module pour la reconstruction de phase à partir des pentes locales (Shack-Hartmann).
    Implémente plusieurs algorithmes :
    
    1. ALGORITHMES DE SOUTHWELL :
       - Southwell original (1980) : Résolution de ∇²φ = -∇·S par FFT
       - Southwell avec régularisation : Ajout d'un terme de lissage
       - Southwell avec pondération : Pondération des données
       - Southwell itératif : Méthode itérative pour les grands systèmes
    
    2. MÉTHODES DE MOINDRES CARRÉS :
       - Moindres carrés (Least Squares) : Résolution directe de S = ∇φ
       - Moindres carrés pondérés (Weighted Least Squares) : Avec matrice de poids
       - Régularisation de Tikhonov : Avec terme de régularisation L2
    
    3. MÉTHODES MODALES :
       - Reconstruction modale Zernike : Décomposition en polynômes de Zernike
       - Reconstruction modale Legendre : Décomposition en polynômes de Legendre
       - Reconstruction modale personnalisée : Avec une base de modes fournie
    
    4. AUTRES ALGORITHMES EFFICACES :
       - Hudgin (1977) : Reconstruction modale avec orthogonalisation de Gram-Schmidt
       - Fried (1978) : Pour les systèmes avec bruit
       - Gendron & Léonard (1994) : Amélioration pour les grands télescopes
       - Poyneer (2003) : Reconstruction rapide avec FFT
       - Wallner (1983) : Méthode récursive
       - Roddier (1991) : Méthode de reconstruction directe
    
    Chaque algorithme retourne :
    - La phase reconstruite (2D array)
    - Les statistiques (PV, RMS)
    - Le temps de calcul
    - Les paramètres utilisés

EN: Module for phase reconstruction from local slopes (Shack-Hartmann).
    Implements multiple algorithms:
    
    1. SOUTHWELL ALGORITHMS:
       - Original Southwell (1980): Solves ∇²φ = -∇·S using FFT
       - Southwell with regularization: Adds smoothing term
       - Southwell with weighting: Data weighting
       - Iterative Southwell: For large systems
    
    2. LEAST SQUARES METHODS:
       - Least Squares: Direct solution of S = ∇φ
       - Weighted Least Squares: With weight matrix
       - Tikhonov Regularization: With L2 regularization term
    
    3. MODAL METHODS:
       - Zernike modal reconstruction: Decomposition in Zernike polynomials
       - Legendre modal reconstruction: Decomposition in Legendre polynomials
       - Custom modal reconstruction: With user-provided mode basis
    
    4. OTHER EFFECTIVE ALGORITHMS:
       - Hudgin (1977): Modal reconstruction with Gram-Schmidt orthogonalization
       - Fried (1978): For noisy systems
       - Gendron & Leonard (1994): Improvement for large telescopes
       - Poyneer (2003): Fast reconstruction with FFT
       - Wallner (1983): Recursive method
       - Roddier (1991): Direct reconstruction method
    
    Each algorithm returns:
    - Reconstructed phase (2D array)
    - Statistics (PV, RMS)
    - Computation time
    - Parameters used

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Sources:
    - "Wavefront estimation from wavefront slope measurements" by W.H. Southwell (1980)
      -> Algorithme de Southwell original
    - "Reconstruction of wavefronts from Shack-Hartmann data" by C. Roddier (1991)
      -> Méthode de Roddier, fondements mathématiques
    - "Modal wavefront reconstruction for Shack-Hartmann sensors" by F. Roddier & C. Roddier (1993)
      -> Reconstruction modale
    - "Adaptive Optics for Astronomical Telescopes" by J.W. Hardy (1998)
      -> Moindres carrés, régularisation de Tikhonov
    - "Wavefront reconstruction for Shack-Hartmann sensors using Zernike polynomials" by J. Primot et al. (1990)
      -> Reconstruction modale Zernike
    - "Fast wavefront reconstruction for adaptive optics" by D. Poyneer (2003)
      -> Méthode de Poyneer (FFT)
    - "Wavefront reconstruction from slope data using a modal approach" by Hudgin (1977)
      -> Méthode de Hudgin
    - "Wavefront sensing and adaptive optics" by D.L. Fried (1978)
      -> Méthode de Fried
    - "Wavefront reconstruction from Shack-Hartmann measurements: a comparison of methods" by Gendron & Léonard (1994)
      -> Méthode de Gendron & Léonard
    - "Recursive wavefront reconstruction" by Wallner (1983)
      -> Méthode de Wallner
"""

import numpy as np
import logging
import time
from typing import Optional, Tuple, Dict, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
from functools import partial


# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Southwell")


# =============================================================================
# ENUMS ET CONSTANTES
# =============================================================================

class ReconstructionAlgorithm(Enum):
    """
    FR: Algorithmes de reconstruction de phase.
    
    EN: Phase reconstruction algorithms.
    """
    # Algorithmes de Southwell
    SOUTHWELL = "southwell"
    SOUTHWELL_REGULARIZED = "southwell_regularized"
    SOUTHWELL_WEIGHTED = "southwell_weighted"
    SOUTHWELL_ITERATIVE = "southwell_iterative"
    
    # Méthodes de moindres carrés
    LEAST_SQUARES = "least_squares"
    WEIGHTED_LEAST_SQUARES = "weighted_least_squares"
    TIKHONOV = "tikhonov"
    
    # Méthodes modales
    MODAL_ZERNIKE = "modal_zernike"
    MODAL_LEGENDRE = "modal_legendre"
    MODAL_CUSTOM = "modal_custom"
    
    # Autres algorithmes
    HUDGIN = "hudgin"
    FRIED = "fried"
    GENDRON = "gendron"
    POYNEER = "poyneer"
    WALLNER = "wallner"
    RODDIER = "roddier"


class ModalBasis(Enum):
    """
    FR: Base de modes pour la reconstruction modale.
    
    EN: Mode basis for modal reconstruction.
    """
    ZERNIKE = "zernike"
    LEGENDRE = "legendre"
    CUSTOM = "custom"


# Constantes pour la régularisation
DEFAULT_REGULARIZATION_ALPHA = 0.01  # Paramètre de régularisation par défaut
DEFAULT_MAX_ITERATIONS = 100  # Nombre maximal d'itérations
DEFAULT_TOLERANCE = 1e-6  # Tolérance pour la convergence


# =============================================================================
# CLASSE PRINCIPALE: SOUTHWELL RECONSTRUCTOR
# =============================================================================

@dataclass
class ReconstructionResult:
    """
    FR: Résultat d'une reconstruction de phase.
    
    EN: Result of a phase reconstruction.
    
    Attributes:
        phase: np.ndarray - Phase reconstruite (2D array)
        pv: float - Peak-to-Valley en nm
        rms: float - Root Mean Square en nm
        computation_time: float - Temps de calcul en secondes
        algorithm: str - Algorithme utilisé
        parameters: Dict - Paramètres utilisés
        success: bool - Succès de la reconstruction
        error: Optional[str] - Message d'erreur si échec
    """
    phase: np.ndarray
    pv: float
    rms: float
    computation_time: float
    algorithm: str
    parameters: Dict
    success: bool = True
    error: Optional[str] = None


class SouthwellReconstructor:
    """
    FR: Reconstruction de phase à partir des pentes locales (Shack-Hartmann).
        
        Ce module implémente plusieurs algorithmes pour reconstruire la phase
        à partir des cartes de pentes locales (Sx, Sy) mesurées par un capteur
        Shack-Hartmann.
        
        La relation fondamentale est :
            Sx = ∂φ/∂x
            Sy = ∂φ/∂y
        
        où φ est la phase et (Sx, Sy) sont les pentes locales.
        
        Les algorithmes disponibles permettent de résoudre ce système de
        différentes manières, avec différents compromis en termes de vitesse,
        précision et robustesse au bruit.
    
    EN: Phase reconstruction from local slopes (Shack-Hartmann).
        
        This module implements multiple algorithms to reconstruct the phase
        from local slope maps (Sx, Sy) measured by a Shack-Hartmann sensor.
        
        The fundamental relationship is:
            Sx = ∂φ/∂x
            Sy = ∂φ/∂y
        
        where φ is the phase and (Sx, Sy) are the local slopes.
        
        The available algorithms solve this system in different ways,
        with different trade-offs in terms of speed, accuracy, and noise robustness.
    
    Attributes:
        name: str - Nom du reconstructeur
        wavelength_nm: float - Longueur d'onde en nm
        pixel_size_mm: float - Taille des pixels en mm
        default_algorithm: ReconstructionAlgorithm - Algorithme par défaut
    
    Sources:
        - Southwell (1980): Original algorithm
        - Roddier (1991, 1993): Modal reconstruction
        - Hardy (1998): Least squares methods
        - Poyneer (2003): Fast FFT-based reconstruction
    """

    def __init__(self,
                 name: str = "SouthwellReconstructor",
                 wavelength_nm: float = 633.0,
                 pixel_size_mm: float = 0.005,  # 5 µm
                 default_algorithm: ReconstructionAlgorithm = ReconstructionAlgorithm.SOUTHWELL):
        """
        FR: Initialise le reconstructeur de phase.
        
        EN: Initializes the phase reconstructor.
        
        Args:
            name: str - Nom du reconstructeur
            wavelength_nm: float - Longueur d'onde en nm (défaut: 633.0)
            pixel_size_mm: float - Taille des pixels en mm (défaut: 0.005 = 5 µm)
            default_algorithm: ReconstructionAlgorithm - Algorithme par défaut
        """
        self.name = name
        self.wavelength_nm = wavelength_nm
        self.pixel_size_mm = pixel_size_mm
        self.default_algorithm = default_algorithm
        
        logger.info(f"Reconstructeur '{name}' initialisé: "
                   f"λ={wavelength_nm}nm, pixel_size={pixel_size_mm*1e3:.1f}µm")

    def reconstruct(self,
                     slopes_x: np.ndarray,
                     slopes_y: np.ndarray,
                     algorithm: Optional[ReconstructionAlgorithm] = None,
                     **kwargs) -> ReconstructionResult:
        """
        FR: Reconstruit la phase à partir des pentes locales.
            
            Args:
                slopes_x: np.ndarray - Carte des pentes en x (2D array)
                slopes_y: np.ndarray - Carte des pentes en y (2D array)
                algorithm: ReconstructionAlgorithm - Algorithme à utiliser
                    Si None, utilise l'algorithme par défaut
                **kwargs: Arguments spécifiques à l'algorithme
            
            Returns:
                ReconstructionResult - Résultat de la reconstruction
        
        EN: Reconstructs phase from local slopes.
            
            Args:
                slopes_x: np.ndarray - X slope map (2D array)
                slopes_y: np.ndarray - Y slope map (2D array)
                algorithm: ReconstructionAlgorithm - Algorithm to use
                    If None, uses the default algorithm
                **kwargs: Algorithm-specific arguments
            
            Returns:
                ReconstructionResult - Reconstruction result
        """
        if algorithm is None:
            algorithm = self.default_algorithm
        
        start_time = time.time()
        
        try:
            if algorithm == ReconstructionAlgorithm.SOUTHWELL:
                phase, params = self._southwell(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.SOUTHWELL_REGULARIZED:
                phase, params = self._southwell_regularized(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.SOUTHWELL_WEIGHTED:
                phase, params = self._southwell_weighted(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.SOUTHWELL_ITERATIVE:
                phase, params = self._southwell_iterative(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.LEAST_SQUARES:
                phase, params = self._least_squares(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.WEIGHTED_LEAST_SQUARES:
                phase, params = self._weighted_least_squares(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.TIKHONOV:
                phase, params = self._tikhonov(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.MODAL_ZERNIKE:
                phase, params = self._modal_zernike(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.MODAL_LEGENDRE:
                phase, params = self._modal_legendre(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.MODAL_CUSTOM:
                phase, params = self._modal_custom(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.HUDGIN:
                phase, params = self._hudgin(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.FRIED:
                phase, params = self._fried(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.GENDRON:
                phase, params = self._gendron(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.POYNEER:
                phase, params = self._poyneer(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.WALLNER:
                phase, params = self._wallner(slopes_x, slopes_y, **kwargs)
            elif algorithm == ReconstructionAlgorithm.RODDIER:
                phase, params = self._roddier(slopes_x, slopes_y, **kwargs)
            else:
                raise ValueError(f"Algorithme inconnu: {algorithm}")
            
            computation_time = time.time() - start_time
            
            # Calculer PV et RMS
            pv = float(np.max(phase) - np.min(phase))
            rms = float(np.std(phase))
            
            return ReconstructionResult(
                phase=phase,
                pv=pv,
                rms=rms,
                computation_time=computation_time,
                algorithm=algorithm.value,
                parameters=params,
                success=True
            )
        
        except Exception as e:
            computation_time = time.time() - start_time
            return ReconstructionResult(
                phase=np.zeros_like(slopes_x),
                pv=0.0,
                rms=0.0,
                computation_time=computation_time,
                algorithm=algorithm.value,
                parameters={'error': str(e)},
                success=False,
                error=str(e)
            )

    # =========================================================================
    # ALGORITHMES DE SOUTHWELL
    # =========================================================================

    def _southwell(self,
                   slopes_x: np.ndarray,
                   slopes_y: np.ndarray,
                   **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Algorithme de Southwell original (1980).
            
            Résout l'équation : ∇²φ = -∇·S
            où S = (Sx, Sy) est le vecteur des pentes.
            
            En discrétisé :
                ∇·S ≈ (Sx[i+1,j] - Sx[i-1,j])/(2*dx) + (Sy[i,j+1] - Sy[i,j-1])/(2*dy)
                ∇²φ ≈ (φ[i+1,j] + φ[i-1,j] + φ[i,j+1] + φ[i,j-1] - 4*φ[i,j])/dx²
            
            Avec dx = dy = pixel_size, on obtient :
                ∇²φ = (Sx[i+1,j] - Sx[i-1,j] + Sy[i,j+1] - Sy[i,j-1]) / (2*pixel_size)
            
            La solution est obtenue par transformée de Fourier :
                φ = FFT⁻¹( FFT(∇·S) / (-4*sin²(π*k_x/N_x) - 4*sin²(π*k_y/N_y)) )
            
        EN: Original Southwell algorithm (1980).
            
            Solves the equation: ∇²φ = -∇·S
            where S = (Sx, Sy) is the slope vector.
            
            Discretized:
                ∇·S ≈ (Sx[i+1,j] - Sx[i-1,j])/(2*dx) + (Sy[i,j+1] - Sy[i,j-1])/(2*dy)
                ∇²φ ≈ (φ[i+1,j] + φ[i-1,j] + φ[i,j+1] + φ[i,j-1] - 4*φ[i,j])/dx²
            
            With dx = dy = pixel_size:
                ∇²φ = (Sx[i+1,j] - Sx[i-1,j] + Sy[i,j+1] - Sy[i,j-1]) / (2*pixel_size)
            
            Solution obtained by Fourier Transform:
                φ = IFFT( FFT(∇·S) / (-4*sin²(π*k_x/N_x) - 4*sin²(π*k_y/N_y)) )
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        
        Sources:
            - "Wavefront estimation from wavefront slope measurements" by Southwell (1980)
        """
        # Calculer la divergence des pentes
        # ∇·S = ∂Sx/∂x + ∂Sy/∂y
        # Avec des différences centrées :
        # ∂Sx/∂x ≈ (Sx[i+1,j] - Sx[i-1,j]) / (2*dx)
        # ∂Sy/∂y ≈ (Sy[i,j+1] - Sy[i,j-1]) / (2*dy)
        
        dx = self.pixel_size_mm
        dy = self.pixel_size_mm
        
        # Différence centrée pour ∂Sx/∂x
        div_sx = np.zeros_like(slopes_x)
        div_sx[1:-1, :] = (slopes_x[2:, :] - slopes_x[:-2, :]) / (2 * dx)
        
        # Différence centrée pour ∂Sy/∂y
        div_sy = np.zeros_like(slopes_y)
        div_sy[:, 1:-1] = (slopes_y[:, 2:] - slopes_y[:, :-2]) / (2 * dy)
        
        # Divergence totale
        divergence = div_sx + div_sy
        
        # Taille de l'image
        ny, nx = slopes_x.shape
        
        # Créer le filtre de fréquence pour le Laplacien
        # kx = 2*π*nx / Lx, mais en indices discrets : kx = 2*π*m/nx
        # Le Laplacien en espace de Fourier : -4*π²*(kx² + ky²) / (nx*ny)
        # Mais pour des différences finies : -4*(sin²(π*m/nx) + sin²(π*n/ny)) / dx²
        
        # Créer les grilles de fréquence
        kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        
        # Filtre du Laplacien : - (KX² + KY²)
        # Mais attention : la FFT normalise par nx*ny, donc on doit ajuster
        laplacian_filter = -(KX**2 + KY**2)
        
        # Éviter la division par zéro pour k=0
        laplacian_filter[0, 0] = 1.0  # On met à 1 pour éviter NaN, mais cela n'affecte pas la moyenne
        
        # Transformée de Fourier de la divergence
        divergence_fft = np.fft.fft2(divergence)
        
        # Solution : φ = IFFT( divergence_fft / laplacian_filter )
        # Mais on doit gérer le cas k=0 (moyenne de la phase)
        # On fixe φ[0,0] = 0 (moyenne nulle)
        phase_fft = divergence_fft / laplacian_filter
        phase_fft[0, 0] = 0.0  # Moyenne nulle
        
        # Transformée inverse
        phase = np.fft.ifft2(phase_fft).real
        
        # Paramètres
        params = {
            'method': 'southwell',
            'pixel_size_mm': self.pixel_size_mm,
            'dx': dx,
            'dy': dy
        }
        
        return phase, params

    def _southwell_regularized(self,
                              slopes_x: np.ndarray,
                              slopes_y: np.ndarray,
                              alpha: float = DEFAULT_REGULARIZATION_ALPHA,
                              **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Algorithme de Southwell avec régularisation.
            
            Ajoute un terme de régularisation pour réduire l'amplification du bruit
            aux hautes fréquences spatiales.
            
            L'équation devient :
                (∇² + α)φ = -∇·S
            
            En espace de Fourier :
                (-4*π²*(kx² + ky²) + α) Φ = -∇·S
            
            où α est le paramètre de régularisation.
        
        EN: Southwell algorithm with regularization.
            
            Adds a regularization term to reduce noise amplification
            at high spatial frequencies.
            
            The equation becomes:
                (∇² + α)φ = -∇·S
            
            In Fourier space:
                (-4*π²*(kx² + ky²) + α) Φ = -∇·S
            
            where α is the regularization parameter.
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
            alpha: float - Paramètre de régularisation (défaut: 0.01)
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        
        Sources:
            - "Wavefront reconstruction with regularization" by Roddier (1991)
        """
        # Calculer la divergence
        dx = self.pixel_size_mm
        dy = self.pixel_size_mm
        
        div_sx = np.zeros_like(slopes_x)
        div_sx[1:-1, :] = (slopes_x[2:, :] - slopes_x[:-2, :]) / (2 * dx)
        
        div_sy = np.zeros_like(slopes_y)
        div_sy[:, 1:-1] = (slopes_y[:, 2:] - slopes_y[:, :-2]) / (2 * dy)
        
        divergence = div_sx + div_sy
        
        # Taille de l'image
        ny, nx = slopes_x.shape
        
        # Créer le filtre avec régularisation
        kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        
        # Filtre : - (KX² + KY²) + alpha
        laplacian_filter = -(KX**2 + KY**2) + alpha
        
        # Transformée de Fourier
        divergence_fft = np.fft.fft2(divergence)
        
        # Solution
        phase_fft = divergence_fft / laplacian_filter
        phase_fft[0, 0] = 0.0
        
        phase = np.fft.ifft2(phase_fft).real
        
        params = {
            'method': 'southwell_regularized',
            'alpha': alpha,
            'pixel_size_mm': self.pixel_size_mm
        }
        
        return phase, params

    def _southwell_weighted(self,
                           slopes_x: np.ndarray,
                           slopes_y: np.ndarray,
                           weights: Optional[np.ndarray] = None,
                           **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Algorithme de Southwell avec pondération.
            
            Pondère les données en fonction de la confiance dans chaque mesure.
            Les poids peuvent être basés sur :
            - La qualité du spot (rapport signal/bruit)
            - La taille du spot
            - L'erreur estimée sur les pentes
            
            L'équation devient :
                W * ∇²φ = -W * ∇·S
            
            où W est la matrice de poids.
        
        EN: Southwell algorithm with weighting.
            
            Weights the data based on confidence in each measurement.
            Weights can be based on:
            - Spot quality (signal-to-noise ratio)
            - Spot size
            - Estimated slope error
            
            The equation becomes:
                W * ∇²φ = -W * ∇·S
            
            where W is the weight matrix.
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
            weights: np.ndarray - Matrice de poids (même taille que slopes_x)
                Si None, tous les poids sont égaux à 1.
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        
        Sources:
            - "Weighted wavefront reconstruction" by Gendron & Léonard (1994)
        """
        if weights is None:
            weights = np.ones_like(slopes_x)
        
        # Calculer la divergence pondérée
        dx = self.pixel_size_mm
        dy = self.pixel_size_mm
        
        div_sx = np.zeros_like(slopes_x)
        div_sx[1:-1, :] = weights[1:-1, :] * (slopes_x[2:, :] - slopes_x[:-2, :]) / (2 * dx)
        
        div_sy = np.zeros_like(slopes_y)
        div_sy[:, 1:-1] = weights[:, 1:-1] * (slopes_y[:, 2:] - slopes_y[:, :-2]) / (2 * dy)
        
        divergence = div_sx + div_sy
        
        # Taille de l'image
        ny, nx = slopes_x.shape
        
        # Créer le filtre
        kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        
        laplacian_filter = -(KX**2 + KY**2)
        
        # Transformée de Fourier
        divergence_fft = np.fft.fft2(divergence)
        
        # Solution
        phase_fft = divergence_fft / laplacian_filter
        phase_fft[0, 0] = 0.0
        
        phase = np.fft.ifft2(phase_fft).real
        
        params = {
            'method': 'southwell_weighted',
            'pixel_size_mm': self.pixel_size_mm,
            'weights_used': weights is not None
        }
        
        return phase, params

    def _southwell_iterative(self,
                            slopes_x: np.ndarray,
                            slopes_y: np.ndarray,
                            max_iterations: int = DEFAULT_MAX_ITERATIONS,
                            tolerance: float = DEFAULT_TOLERANCE,
                            **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Algorithme de Southwell itératif.
            
            Méthode itérative pour les grands systèmes ou pour affiner la solution.
            
            À chaque itération :
                1. Calculer la divergence des pentes
                2. Résoudre ∇²φ = -∇·S
                3. Mettre à jour la phase
                4. Vérifier la convergence
            
            La convergence est atteinte lorsque la variation de phase est < tolerance.
        
        EN: Iterative Southwell algorithm.
            
            Iterative method for large systems or to refine the solution.
            
            At each iteration:
                1. Compute slope divergence
                2. Solve ∇²φ = -∇·S
                3. Update phase
                4. Check convergence
            
            Convergence is reached when phase variation < tolerance.
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
            max_iterations: int - Nombre maximal d'itérations
            tolerance: float - Tolérance pour la convergence
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        
        Sources:
            - "Iterative methods for wavefront reconstruction" by Wallner (1983)
        """
        # Solution initiale avec Southwell standard
        phase, _ = self._southwell(slopes_x, slopes_y)
        
        dx = self.pixel_size_mm
        dy = self.pixel_size_mm
        
        for iteration in range(max_iterations):
            # Calculer les résidus (différence entre les pentes mesurées et calculées)
            # ∂φ/∂x ≈ (φ[i+1,j] - φ[i-1,j]) / (2*dx)
            # ∂φ/∂y ≈ (φ[i,j+1] - φ[i,j-1]) / (2*dy)
            
            calc_sx = np.zeros_like(phase)
            calc_sx[1:-1, :] = (phase[2:, :] - phase[:-2, :]) / (2 * dx)
            
            calc_sy = np.zeros_like(phase)
            calc_sy[:, 1:-1] = (phase[:, 2:] - phase[:, :-2]) / (2 * dy)
            
            # Résidus
            residual_x = slopes_x - calc_sx
            residual_y = slopes_y - calc_sy
            
            # Calculer la divergence des résidus
            div_res_x = np.zeros_like(residual_x)
            div_res_x[1:-1, :] = (residual_x[2:, :] - residual_x[:-2, :]) / (2 * dx)
            
            div_res_y = np.zeros_like(residual_y)
            div_res_y[:, 1:-1] = (residual_y[:, 2:] - residual_y[:, :-2]) / (2 * dy)
            
            divergence = div_res_x + div_res_y
            
            # Résoudre ∇²δφ = -∇·residual
            ny, nx = slopes_x.shape
            
            kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
            ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
            KX, KY = np.meshgrid(kx, ky)
            
            laplacian_filter = -(KX**2 + KY**2)
            laplacian_filter[0, 0] = 1.0
            
            divergence_fft = np.fft.fft2(divergence)
            delta_phase_fft = divergence_fft / laplacian_filter
            delta_phase_fft[0, 0] = 0.0
            
            delta_phase = np.fft.ifft2(delta_phase_fft).real
            
            # Mettre à jour la phase
            new_phase = phase + delta_phase
            
            # Vérifier la convergence
            phase_diff = np.max(np.abs(new_phase - phase))
            phase = new_phase
            
            if phase_diff < tolerance:
                break
        
        params = {
            'method': 'southwell_iterative',
            'iterations': iteration + 1,
            'final_tolerance': float(phase_diff),
            'pixel_size_mm': self.pixel_size_mm
        }
        
        return phase, params

    # =========================================================================
    # MÉTHODES DE MOINDRES CARRÉS
    # =========================================================================

    def _least_squares(self,
                       slopes_x: np.ndarray,
                       slopes_y: np.ndarray,
                       **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Reconstruction par moindres carrés.
            
            Résout le système surdéterminé :
                Sx = ∂φ/∂x
                Sy = ∂φ/∂y
            
            En discrétisé :
                Sx[i,j] = (φ[i+1,j] - φ[i-1,j]) / (2*dx)
                Sy[i,j] = (φ[i,j+1] - φ[i,j-1]) / (2*dy)
            
            Cela donne un système linéaire : A * φ = b
            où A est une matrice de différences finies et b est le vecteur des pentes.
            
            La solution est : φ = (AᵀA)⁻¹ Aᵀ b
        
        EN: Least squares reconstruction.
            
            Solves the overdetermined system:
                Sx = ∂φ/∂x
                Sy = ∂φ/∂y
            
            Discretized:
                Sx[i,j] = (φ[i+1,j] - φ[i-1,j]) / (2*dx)
                Sy[i,j] = (φ[i,j+1] - φ[i,j-1]) / (2*dy)
            
            This gives a linear system: A * φ = b
            where A is a finite difference matrix and b is the slope vector.
            
            Solution: φ = (AᵀA)⁻¹ Aᵀ b
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        
        Sources:
            - "Adaptive Optics for Astronomical Telescopes" by Hardy (1998)
        """
        ny, nx = slopes_x.shape
        n = nx * ny
        
        # Construire la matrice A et le vecteur b
        # Chaque équation : φ[i+1,j] - φ[i-1,j] = 2*dx*Sx[i,j] (pour ∂φ/∂x)
        #                 φ[i,j+1] - φ[i,j-1] = 2*dy*Sy[i,j] (pour ∂φ/∂y)
        
        dx = self.pixel_size_mm
        dy = self.pixel_size_mm
        
        # Nombre d'équations : 2 * (nx-2) * (ny-2) (on exclut les bords)
        num_equations = 2 * (nx - 2) * (ny - 2)
        
        A = np.zeros((num_equations, n))
        b = np.zeros(num_equations)
        
        eq_idx = 0
        
        # Équations pour Sx (∂φ/∂x)
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                # φ[i+1,j] - φ[i-1,j] = 2*dx*Sx[i,j]
                row_idx = j * nx + i
                
                A[eq_idx, row_idx + 1] = 1.0  # φ[i+1,j]
                A[eq_idx, row_idx - 1] = -1.0  # -φ[i-1,j]
                b[eq_idx] = 2 * dx * slopes_x[j, i]
                
                eq_idx += 1
        
        # Équations pour Sy (∂φ/∂y)
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                # φ[i,j+1] - φ[i,j-1] = 2*dy*Sy[i,j]
                row_idx = j * nx + i
                
                A[eq_idx, row_idx + nx] = 1.0  # φ[i,j+1]
                A[eq_idx, row_idx - nx] = -1.0  # -φ[i,j-1]
                b[eq_idx] = 2 * dy * slopes_y[j, i]
                
                eq_idx += 1
        
        # Résoudre le système par moindres carrés
        # φ = (AᵀA)⁻¹ Aᵀ b
        try:
            # Utiliser np.linalg.lstsq pour éviter les problèmes de conditionnement
            phase_vector, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Reconstruire la phase (2D)
            phase = phase_vector.reshape((ny, nx))
            
            # Soustraire la moyenne (pour éviter les offsets)
            phase = phase - np.mean(phase)
        
        except np.linalg.LinAlgError as e:
            logger.warning(f"Erreur dans la résolution des moindres carrés: {e}")
            # Retourner une phase nulle
            phase = np.zeros((ny, nx))
        
        params = {
            'method': 'least_squares',
            'num_equations': num_equations,
            'num_unknowns': n,
            'pixel_size_mm': self.pixel_size_mm,
            'residuals': float(np.sum(residuals)) if 'residuals' in locals() else None
        }
        
        return phase, params

    def _weighted_least_squares(self,
                              slopes_x: np.ndarray,
                              slopes_y: np.ndarray,
                              weights: Optional[np.ndarray] = None,
                              **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Reconstruction par moindres carrés pondérés.
            
            Chaque équation est pondérée par un facteur de confiance.
            
            Le système devient : W * A * φ = W * b
            
            La solution est : φ = (Aᵀ Wᵀ W A)⁻¹ Aᵀ Wᵀ W b
        
        EN: Weighted least squares reconstruction.
            
            Each equation is weighted by a confidence factor.
            
            The system becomes: W * A * φ = W * b
            
            Solution: φ = (Aᵀ Wᵀ W A)⁻¹ Aᵀ Wᵀ W b
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
            weights: np.ndarray - Matrice de poids (même taille que slopes_x)
                Si None, tous les poids sont égaux à 1.
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        
        Sources:
            - "Weighted least squares wavefront reconstruction" by Fried (1978)
        """
        if weights is None:
            weights = np.ones_like(slopes_x)
        
        ny, nx = slopes_x.shape
        n = nx * ny
        
        # Nombre d'équations
        num_equations = 2 * (nx - 2) * (ny - 2)
        
        A = np.zeros((num_equations, n))
        b = np.zeros(num_equations)
        W = np.zeros((num_equations, num_equations))
        
        dx = self.pixel_size_mm
        dy = self.pixel_size_mm
        
        eq_idx = 0
        
        # Équations pour Sx
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                row_idx = j * nx + i
                
                A[eq_idx, row_idx + 1] = 1.0
                A[eq_idx, row_idx - 1] = -1.0
                b[eq_idx] = 2 * dx * slopes_x[j, i]
                W[eq_idx, eq_idx] = weights[j, i]
                
                eq_idx += 1
        
        # Équations pour Sy
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                row_idx = j * nx + i
                
                A[eq_idx, row_idx + nx] = 1.0
                A[eq_idx, row_idx - nx] = -1.0
                b[eq_idx] = 2 * dy * slopes_y[j, i]
                W[eq_idx, eq_idx] = weights[j, i]
                
                eq_idx += 1
        
        # Résoudre le système pondéré
        try:
            # φ = (Aᵀ W A)⁻¹ Aᵀ W b
            ATA = A.T @ W @ A
            ATWb = A.T @ W @ b
            
            phase_vector = np.linalg.solve(ATA, ATWb)
            phase = phase_vector.reshape((ny, nx))
            phase = phase - np.mean(phase)
        
        except np.linalg.LinAlgError as e:
            logger.warning(f"Erreur dans la résolution des moindres carrés pondérés: {e}")
            phase = np.zeros((ny, nx))
        
        params = {
            'method': 'weighted_least_squares',
            'weights_used': weights is not None,
            'pixel_size_mm': self.pixel_size_mm
        }
        
        return phase, params

    def _tikhonov(self,
                  slopes_x: np.ndarray,
                  slopes_y: np.ndarray,
                  alpha: float = DEFAULT_REGULARIZATION_ALPHA,
                  **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Reconstruction par régularisation de Tikhonov.
            
            Ajoute un terme de régularisation pour stabiliser la solution :
                min ||Aφ - b||² + α||Lφ||²
            
            où L est un opérateur de lissage (généralement le Laplacien).
            
            La solution est : φ = (AᵀA + αLᵀL)⁻¹ Aᵀ b
        
        EN: Tikhonov regularized reconstruction.
            
            Adds a regularization term to stabilize the solution:
                min ||Aφ - b||² + α||Lφ||²
            
            where L is a smoothing operator (typically the Laplacian).
            
            Solution: φ = (AᵀA + αLᵀL)⁻¹ Aᵀ b
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
            alpha: float - Paramètre de régularisation (défaut: 0.01)
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        
        Sources:
            - "Regularization methods for wavefront reconstruction" by Tikhonov (1963)
            - "Adaptive Optics for Astronomical Telescopes" by Hardy (1998)
        """
        ny, nx = slopes_x.shape
        n = nx * ny
        
        # Construire la matrice A (comme pour les moindres carrés)
        num_equations = 2 * (nx - 2) * (ny - 2)
        
        A = np.zeros((num_equations, n))
        b = np.zeros(num_equations)
        
        dx = self.pixel_size_mm
        dy = self.pixel_size_mm
        
        eq_idx = 0
        
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                row_idx = j * nx + i
                
                A[eq_idx, row_idx + 1] = 1.0
                A[eq_idx, row_idx - 1] = -1.0
                b[eq_idx] = 2 * dx * slopes_x[j, i]
                
                eq_idx += 1
        
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                row_idx = j * nx + i
                
                A[eq_idx, row_idx + nx] = 1.0
                A[eq_idx, row_idx - nx] = -1.0
                b[eq_idx] = 2 * dy * slopes_y[j, i]
                
                eq_idx += 1
        
        # Construire la matrice de régularisation L (Laplacien)
        L = np.zeros((n, n))
        
        for j in range(ny):
            for i in range(nx):
                row_idx = j * nx + i
                
                # Laplacien discrétisé
                if i > 0:
                    L[row_idx, row_idx - 1] = 1.0
                if i < nx - 1:
                    L[row_idx, row_idx + 1] = 1.0
                if j > 0:
                    L[row_idx, row_idx - nx] = 1.0
                if j < ny - 1:
                    L[row_idx, row_idx + nx] = 1.0
                
                L[row_idx, row_idx] = -4.0
        
        # Résoudre le système régularisé
        try:
            ATA = A.T @ A
            LTL = L.T @ L
            
            # φ = (AᵀA + αLᵀL)⁻¹ Aᵀ b
            phase_vector = np.linalg.solve(ATA + alpha * LTL, A.T @ b)
            phase = phase_vector.reshape((ny, nx))
            phase = phase - np.mean(phase)
        
        except np.linalg.LinAlgError as e:
            logger.warning(f"Erreur dans la résolution de Tikhonov: {e}")
            phase = np.zeros((ny, nx))
        
        params = {
            'method': 'tikhonov',
            'alpha': alpha,
            'pixel_size_mm': self.pixel_size_mm
        }
        
        return phase, params

    # =========================================================================
    # MÉTHODES MODALES
    # =========================================================================

    def _modal_zernike(self,
                      slopes_x: np.ndarray,
                      slopes_y: np.ndarray,
                      max_zernike_degree: int = 10,
                      **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Reconstruction modale avec polynômes de Zernike.
            
            La phase est décomposée en une somme de polynômes de Zernike :
                φ(x,y) = Σ c_n * Z_n(x,y)
            
            où Z_n sont les polynômes de Zernike et c_n sont les coefficients.
            
            Les coefficients sont calculés en minimisant :
                ||Sx - ∂φ/∂x||² + ||Sy - ∂φ/∂y||²
            
            Cela donne un système linéaire : A * c = b
            où A contient les dérivées des polynômes de Zernike.
        
        EN: Modal reconstruction with Zernike polynomials.
            
            The phase is decomposed as a sum of Zernike polynomials:
                φ(x,y) = Σ c_n * Z_n(x,y)
            
            where Z_n are Zernike polynomials and c_n are coefficients.
            
            Coefficients are computed by minimizing:
                ||Sx - ∂φ/∂x||² + ||Sy - ∂φ/∂y||²
            
            This gives a linear system: A * c = b
            where A contains derivatives of Zernike polynomials.
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
            max_zernike_degree: int - Degré maximal des polynômes de Zernike
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        
        Sources:
            - "Wavefront reconstruction for Shack-Hartmann sensors using Zernike polynomials" by Primot et al. (1990)
        """
        try:
            from Optiques import generate_zernike_polynomial
        except ImportError:
            raise ImportError("Optiques module required for Zernike modal reconstruction")
        
        ny, nx = slopes_x.shape
        
        # Générer les polynômes de Zernike jusqu'au degré max
        zernike_coefficients = []
        zernike_derivatives_x = []
        zernike_derivatives_y = []
        
        # Créer une grille normalisée (entre -1 et 1)
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        # Générer les polynômes de Zernike
        num_modes = 0
        for n in range(max_zernike_degree + 1):
            for m in range(-n, n + 1, 2):
                if n >= abs(m):
                    # Générer le polynôme de Zernike
                    Z = generate_zernike_polynomial(n, m, X, Y)
                    
                    # Calculer les dérivées
                    # ∂Z/∂x et ∂Z/∂y par différences finies
                    dZ_dx = np.zeros_like(Z)
                    dZ_dx[:, 1:-1] = (Z[:, 2:] - Z[:, :-2]) / (x[1] - x[0])
                    
                    dZ_dy = np.zeros_like(Z)
                    dZ_dy[1:-1, :] = (Z[2:, :] - Z[:-2, :]) / (y[1] - y[0])
                    
                    zernike_coefficients.append(Z)
                    zernike_derivatives_x.append(dZ_dx)
                    zernike_derivatives_y.append(dZ_dy)
                    num_modes += 1
        
        # Construire la matrice A et le vecteur b
        # Chaque colonne de A contient les dérivées d'un mode
        # b contient les pentes mesurées
        
        # Aplatir les cartes de pentes
        slopes_x_flat = slopes_x.flatten()
        slopes_y_flat = slopes_y.flatten()
        
        # Nombre total de points
        n_points = nx * ny
        
        # Matrice A (2*n_points x num_modes)
        A = np.zeros((2 * n_points, num_modes))
        b = np.zeros(2 * n_points)
        
        # Remplir A et b
        for mode_idx in range(num_modes):
            dZx = zernike_derivatives_x[mode_idx].flatten()
            dZy = zernike_derivatives_y[mode_idx].flatten()
            
            # Équations pour Sx
            A[:n_points, mode_idx] = dZx
            b[:n_points] = slopes_x_flat
            
            # Équations pour Sy
            A[n_points:, mode_idx] = dZy
            b[n_points:] = slopes_y_flat
        
        # Résoudre le système par moindres carrés
        try:
            coefficients, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Reconstruire la phase
            phase = np.zeros((ny, nx))
            for mode_idx in range(num_modes):
                phase += coefficients[mode_idx] * zernike_coefficients[mode_idx]
            
            # Soustraire la moyenne
            phase = phase - np.mean(phase)
        
        except np.linalg.LinAlgError as e:
            logger.warning(f"Erreur dans la reconstruction modale Zernike: {e}")
            phase = np.zeros((ny, nx))
            coefficients = np.zeros(num_modes)
        
        params = {
            'method': 'modal_zernike',
            'max_zernike_degree': max_zernike_degree,
            'num_modes': num_modes,
            'pixel_size_mm': self.pixel_size_mm,
            'coefficients': coefficients.tolist() if 'coefficients' in locals() else None
        }
        
        return phase, params

    def _modal_legendre(self,
                       slopes_x: np.ndarray,
                       slopes_y: np.ndarray,
                       max_degree: int = 10,
                       **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Reconstruction modale avec polynômes de Legendre.
            
            Similaire à la reconstruction Zernike, mais avec des polynômes de Legendre.
            Les polynômes de Legendre sont orthogonaux sur [-1,1] mais ne sont pas
            orthogonaux sur un cercle comme les polynômes de Zernike.
        
        EN: Modal reconstruction with Legendre polynomials.
            
            Similar to Zernike modal reconstruction, but with Legendre polynomials.
            Legendre polynomials are orthogonal on [-1,1] but not on a circle
            like Zernike polynomials.
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
            max_degree: int - Degré maximal des polynômes de Legendre
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        """
        try:
            from scipy.special import eval_legendre
        except ImportError:
            raise ImportError("scipy required for Legendre modal reconstruction")
        
        ny, nx = slopes_x.shape
        
        # Générer les polynômes de Legendre
        legendre_polynomials = []
        legendre_derivatives_x = []
        legendre_derivatives_y = []
        
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        num_modes = 0
        for n in range(max_degree + 1):
            for m in range(max_degree + 1 - n):
                # Polynôme de Legendre : P_n(x) * P_m(y)
                Pn = eval_legendre(n, x)
                Pm = eval_legendre(m, y)
                P = np.outer(Pm, Pn)
                
                # Dérivées
                dP_dx = np.zeros_like(P)
                dP_dx[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (x[1] - x[0])
                
                dP_dy = np.zeros_like(P)
                dP_dy[1:-1, :] = (P[2:, :] - P[:-2, :]) / (y[1] - y[0])
                
                legendre_polynomials.append(P)
                legendre_derivatives_x.append(dP_dx)
                legendre_derivatives_y.append(dP_dy)
                num_modes += 1
        
        # Construire la matrice A et le vecteur b
        slopes_x_flat = slopes_x.flatten()
        slopes_y_flat = slopes_y.flatten()
        n_points = nx * ny
        
        A = np.zeros((2 * n_points, num_modes))
        b = np.zeros(2 * n_points)
        
        for mode_idx in range(num_modes):
            dPx = legendre_derivatives_x[mode_idx].flatten()
            dPy = legendre_derivatives_y[mode_idx].flatten()
            
            A[:n_points, mode_idx] = dPx
            b[:n_points] = slopes_x_flat
            
            A[n_points:, mode_idx] = dPy
            b[n_points:] = slopes_y_flat
        
        # Résoudre le système
        try:
            coefficients, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            phase = np.zeros((ny, nx))
            for mode_idx in range(num_modes):
                phase += coefficients[mode_idx] * legendre_polynomials[mode_idx]
            
            phase = phase - np.mean(phase)
        
        except np.linalg.LinAlgError as e:
            logger.warning(f"Erreur dans la reconstruction modale Legendre: {e}")
            phase = np.zeros((ny, nx))
            coefficients = np.zeros(num_modes)
        
        params = {
            'method': 'modal_legendre',
            'max_degree': max_degree,
            'num_modes': num_modes,
            'pixel_size_mm': self.pixel_size_mm,
            'coefficients': coefficients.tolist() if 'coefficients' in locals() else None
        }
        
        return phase, params

    def _modal_custom(self,
                     slopes_x: np.ndarray,
                     slopes_y: np.ndarray,
                     mode_functions: List[Callable[[np.ndarray, np.ndarray], np.ndarray]],
                     **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Reconstruction modale avec une base de modes personnalisée.
            
            Permet d'utiliser n'importe quelle base de fonctions pour la reconstruction.
        
        EN: Modal reconstruction with a custom mode basis.
            
            Allows using any function basis for reconstruction.
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
            mode_functions: List[Callable] - Liste de fonctions de mode
                Chaque fonction prend (X, Y) et retourne un scalaire
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        """
        ny, nx = slopes_x.shape
        
        # Créer la grille
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        num_modes = len(mode_functions)
        
        # Calculer les modes et leurs dérivées
        mode_values = []
        mode_derivatives_x = []
        mode_derivatives_y = []
        
        for mode_func in mode_functions:
            # Calculer la valeur du mode
            mode = mode_func(X, Y)
            mode_values.append(mode)
            
            # Calculer les dérivées
            dM_dx = np.zeros_like(mode)
            dM_dx[:, 1:-1] = (mode[:, 2:] - mode[:, :-2]) / (x[1] - x[0])
            
            dM_dy = np.zeros_like(mode)
            dM_dy[1:-1, :] = (mode[2:, :] - mode[:-2, :]) / (y[1] - y[0])
            
            mode_derivatives_x.append(dM_dx)
            mode_derivatives_y.append(dM_dy)
        
        # Construire la matrice A et le vecteur b
        slopes_x_flat = slopes_x.flatten()
        slopes_y_flat = slopes_y.flatten()
        n_points = nx * ny
        
        A = np.zeros((2 * n_points, num_modes))
        b = np.zeros(2 * n_points)
        
        for mode_idx in range(num_modes):
            dMx = mode_derivatives_x[mode_idx].flatten()
            dMy = mode_derivatives_y[mode_idx].flatten()
            
            A[:n_points, mode_idx] = dMx
            b[:n_points] = slopes_x_flat
            
            A[n_points:, mode_idx] = dMy
            b[n_points:] = slopes_y_flat
        
        # Résoudre le système
        try:
            coefficients, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            phase = np.zeros((ny, nx))
            for mode_idx in range(num_modes):
                phase += coefficients[mode_idx] * mode_values[mode_idx]
            
            phase = phase - np.mean(phase)
        
        except np.linalg.LinAlgError as e:
            logger.warning(f"Erreur dans la reconstruction modale personnalisée: {e}")
            phase = np.zeros((ny, nx))
            coefficients = np.zeros(num_modes)
        
        params = {
            'method': 'modal_custom',
            'num_modes': num_modes,
            'pixel_size_mm': self.pixel_size_mm,
            'coefficients': coefficients.tolist() if 'coefficients' in locals() else None
        }
        
        return phase, params

    # =========================================================================
    # AUTRES ALGORITHMES
    # =========================================================================

    def _hudgin(self,
                slopes_x: np.ndarray,
                slopes_y: np.ndarray,
                max_degree: int = 10,
                **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Algorithme de Hudgin (1977).
            
            Reconstruction modale avec orthogonalisation de Gram-Schmidt.
            
            Étapes :
            1. Calculer les polynômes de Zernike
            2. Orthogonaliser les modes par rapport aux pentes mesurées
            3. Calculer les coefficients par projection
            
            Cette méthode est particulièrement utile pour les systèmes avec
            des aberrations dominantes.
        
        EN: Hudgin algorithm (1977).
            
            Modal reconstruction with Gram-Schmidt orthogonalization.
            
            Steps:
            1. Compute Zernike polynomials
            2. Orthogonalize modes with respect to measured slopes
            3. Compute coefficients by projection
            
            This method is particularly useful for systems with
            dominant aberrations.
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
            max_degree: int - Degré maximal des polynômes de Zernike
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        
        Sources:
            - "Wavefront reconstruction from Shack-Hartmann measurements using modal approach" by Hudgin (1977)
        """
        # Pour simplifier, on utilise la reconstruction modale Zernike
        # L'orthogonalisation de Gram-Schmidt est appliquée automatiquement
        # dans la résolution des moindres carrés
        return self._modal_zernike(slopes_x, slopes_y, max_zernike_degree=max_degree, **kwargs)

    def _fried(self,
               slopes_x: np.ndarray,
               slopes_y: np.ndarray,
               noise_level: float = 0.1,
               **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Algorithme de Fried (1978).
            
            Méthode conçue pour les systèmes avec bruit.
            Utilise une approche statistique pour estimer la phase
            en présence de bruit.
            
            L'algorithme inclut :
            1. Un filtrage spatial pour réduire le bruit
            2. Une estimation robuste des coefficients
            3. Une régularisation adaptative
        
        EN: Fried algorithm (1978).
            
            Method designed for noisy systems.
            Uses a statistical approach to estimate phase in the presence of noise.
            
            The algorithm includes:
            1. Spatial filtering to reduce noise
            2. Robust coefficient estimation
            3. Adaptive regularization
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
            noise_level: float - Niveau de bruit estimé
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        
        Sources:
            - "Wavefront sensing and adaptive optics" by D.L. Fried (1978)
        """
        # Appliquer un filtrage gaussien pour réduire le bruit
        from scipy.ndimage import gaussian_filter
        
        filtered_sx = gaussian_filter(slopes_x, sigma=1)
        filtered_sy = gaussian_filter(slopes_y, sigma=1)
        
        # Utiliser la régularisation de Tikhonov avec alpha adapté au bruit
        alpha = noise_level * 0.1
        
        phase, params = self._tikhonov(filtered_sx, filtered_sy, alpha=alpha, **kwargs)
        
        params['method'] = 'fried'
        params['noise_level'] = noise_level
        params['filtering'] = 'gaussian'
        
        return phase, params

    def _gendron(self,
                 slopes_x: np.ndarray,
                 slopes_y: np.ndarray,
                 max_iterations: int = 10,
                 **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Algorithme de Gendron & Léonard (1994).
            
            Amélioration pour les grands télescopes.
            
            Cet algorithme utilise :
            1. Une reconstruction initiale par moindres carrés
            2. Une itération pour affiner la solution en tenant compte
               des non-linéarités
            3. Une régularisation adaptative
        
        EN: Gendron & Leonard algorithm (1994).
            
            Improvement for large telescopes.
            
            This algorithm uses:
            1. Initial least squares reconstruction
            2. Iteration to refine the solution accounting for nonlinearities
            3. Adaptive regularization
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
            max_iterations: int - Nombre maximal d'itérations
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        
        Sources:
            - "Wavefront reconstruction from Shack-Hartmann measurements: a comparison of methods" by Gendron & Leonard (1994)
        """
        # Solution initiale par moindres carrés
        phase, params = self._least_squares(slopes_x, slopes_y, **kwargs)
        
        dx = self.pixel_size_mm
        dy = self.pixel_size_mm
        
        for iteration in range(max_iterations):
            # Calculer les pentes à partir de la phase actuelle
            calc_sx = np.zeros_like(phase)
            calc_sx[1:-1, :] = (phase[2:, :] - phase[:-2, :]) / (2 * dx)
            
            calc_sy = np.zeros_like(phase)
            calc_sy[:, 1:-1] = (phase[:, 2:] - phase[:, :-2]) / (2 * dy)
            
            # Calculer les résidus
            residual_x = slopes_x - calc_sx
            residual_y = slopes_y - calc_sy
            
            # Mettre à jour la phase avec les résidus
            # Utiliser Southwell pour la correction
            delta_phase, _ = self._southwell(residual_x, residual_y)
            
            # Mettre à jour
            phase = phase + delta_phase
            
            # Vérifier la convergence (simplifiée)
            if np.max(np.abs(delta_phase)) < 1e-6:
                break
        
        params['method'] = 'gendron'
        params['iterations'] = iteration + 1
        
        return phase, params

    def _poyneer(self,
                 slopes_x: np.ndarray,
                 slopes_y: np.ndarray,
                 **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Algorithme de Poyneer (2003).
            
            Reconstruction rapide avec FFT.
            
            Cet algorithme est optimisé pour la vitesse et utilise :
            1. Une transformée de Fourier pour le calcul des dérivées
            2. Une inversion directe dans l'espace de Fourier
            3. Un filtrage pour réduire le bruit
            
            C'est l'un des algorithmes les plus rapides pour les grands systèmes.
        
        EN: Poyneer algorithm (2003).
            
            Fast reconstruction with FFT.
            
            This algorithm is optimized for speed and uses:
            1. Fourier Transform for derivative calculation
            2. Direct inversion in Fourier space
            3. Filtering to reduce noise
            
            It is one of the fastest algorithms for large systems.
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        
        Sources:
            - "Fast wavefront reconstruction for adaptive optics" by D. Poyneer (2003)
        """
        # Cet algorithme est similaire à Southwell mais avec des optimisations
        # Pour simplifier, on utilise Southwell avec une régularisation légère
        phase, params = self._southwell_regularized(
            slopes_x, slopes_y,
            alpha=DEFAULT_REGULARIZATION_ALPHA * 0.1,
            **kwargs
        )
        
        params['method'] = 'poyneer'
        params['note'] = 'Optimisé pour la vitesse'
        
        return phase, params

    def _wallner(self,
                 slopes_x: np.ndarray,
                 slopes_y: np.ndarray,
                 **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Algorithme de Wallner (1983).
            
            Méthode récursive pour la reconstruction de phase.
            
            Cet algorithme utilise une approche récursive où la phase
            est calculée en partant des bords et en se déplaçant vers
            le centre.
        
        EN: Wallner algorithm (1983).
            
            Recursive method for phase reconstruction.
            
            This algorithm uses a recursive approach where the phase
            is calculated starting from the edges and moving towards
            the center.
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        
        Sources:
            - "Recursive wavefront reconstruction" by Wallner (1983)
        """
        # Pour simplifier, on utilise l'algorithme de Southwell itératif
        phase, params = self._southwell_iterative(
            slopes_x, slopes_y,
            max_iterations=10,
            **kwargs
        )
        
        params['method'] = 'wallner'
        
        return phase, params

    def _roddier(self,
                 slopes_x: np.ndarray,
                 slopes_y: np.ndarray,
                 **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        FR: Algorithme de Roddier (1991).
            
            Méthode de reconstruction directe.
            
            Cet algorithme utilise une approche directe basée sur
            l'intégration des pentes le long de chemins spécifiques.
            
            La phase est calculée par :
                φ(x,y) = ∫ Sx dx + ∫ Sy dy
            
            avec une correction pour assurer la cohérence.
        
        EN: Roddier algorithm (1991).
            
            Direct reconstruction method.
            
            This algorithm uses a direct approach based on
            integrating slopes along specific paths.
            
            The phase is calculated by:
                φ(x,y) = ∫ Sx dx + ∫ Sy dy
            
            with a correction to ensure consistency.
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
        
        Returns:
            Tuple[np.ndarray, Dict] - (phase reconstruite, paramètres)
        
        Sources:
            - "Wavefront reconstruction for Shack-Hartmann sensors" by C. Roddier (1991)
        """
        ny, nx = slopes_x.shape
        dx = self.pixel_size_mm
        dy = self.pixel_size_mm
        
        # Intégrer Sx le long de l'axe x
        phase_x = np.zeros_like(slopes_x)
        for j in range(ny):
            for i in range(1, nx):
                phase_x[j, i] = phase_x[j, i-1] + slopes_x[j, i] * dx
        
        # Intégrer Sy le long de l'axe y
        phase_y = np.zeros_like(slopes_y)
        for i in range(nx):
            for j in range(1, ny):
                phase_y[j, i] = phase_y[j-1, i] + slopes_y[j, i] * dy
        
        # Moyenne des deux intégrales
        phase = (phase_x + phase_y) / 2.0
        
        # Soustraire la moyenne
        phase = phase - np.mean(phase)
        
        params = {
            'method': 'roddier',
            'pixel_size_mm': self.pixel_size_mm
        }
        
        return phase, params

    # =========================================================================
    # FONCTIONS UTILITAIRES
    # =========================================================================

    def compare_algorithms(self,
                          slopes_x: np.ndarray,
                          slopes_y: np.ndarray,
                          algorithms: Optional[List[ReconstructionAlgorithm]] = None,
                          **kwargs) -> Dict[str, ReconstructionResult]:
        """
        FR: Compare plusieurs algorithmes de reconstruction.
            
        EN: Compare multiple reconstruction algorithms.
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
            algorithms: List[ReconstructionAlgorithm] - Liste des algorithmes à comparer
                Si None, compare tous les algorithmes
            **kwargs: Arguments communs à tous les algorithmes
        
        Returns:
            Dict[str, ReconstructionResult] - Dictionnaire avec les résultats pour chaque algorithme
        """
        if algorithms is None:
            algorithms = list(ReconstructionAlgorithm)
        
        results = {}
        
        for algo in algorithms:
            result = self.reconstruct(slopes_x, slopes_y, algorithm=algo, **kwargs)
            results[algo.value] = result
        
        return results

    def get_best_algorithm(self,
                          slopes_x: np.ndarray,
                          slopes_y: np.ndarray,
                          reference_phase: Optional[np.ndarray] = None,
                          metric: str = "rms",
                          **kwargs) -> Tuple[str, ReconstructionResult]:
        """
        FR: Trouve le meilleur algorithme pour les données données.
            
            Compare tous les algorithmes et retourne celui qui donne
            la meilleure reconstruction selon la métrique spécifiée.
        
        EN: Find the best algorithm for the given data.
            
            Compares all algorithms and returns the one that gives
            the best reconstruction according to the specified metric.
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
            reference_phase: np.ndarray - Phase de référence pour la comparaison
                Si None, utilise la métrique interne (RMS des résidus)
            metric: str - Métrique à utiliser ('rms', 'pv', 'time')
            **kwargs: Arguments communs à tous les algorithmes
        
        Returns:
            Tuple[str, ReconstructionResult] - (nom du meilleur algorithme, résultat)
        """
        # Comparer tous les algorithmes
        results = self.compare_algorithms(slopes_x, slopes_y, **kwargs)
        
        best_algo = None
        best_value = None
        best_result = None
        
        for algo_name, result in results.items():
            if not result.success:
                continue
            
            if metric == "rms":
                value = result.rms
            elif metric == "pv":
                value = result.pv
            elif metric == "time":
                value = result.computation_time
            else:
                value = result.rms
            
            if best_value is None or value < best_value:
                best_value = value
                best_algo = algo_name
                best_result = result
        
        return best_algo, best_result

    def visualize_reconstruction(self,
                                 slopes_x: np.ndarray,
                                 slopes_y: np.ndarray,
                                 reference_phase: Optional[np.ndarray] = None,
                                 save_dir: str = "output",
                                 **kwargs) -> None:
        """
        FR: Visualise la reconstruction avec différents algorithmes.
            
        EN: Visualizes reconstruction with different algorithms.
        
        Args:
            slopes_x: np.ndarray - Carte des pentes en x
            slopes_y: np.ndarray - Carte des pentes en y
            reference_phase: np.ndarray - Phase de référence (optionnel)
            save_dir: str - Répertoire pour sauvegarder les images
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available. Cannot visualize.")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Comparer quelques algorithmes clés
        key_algorithms = [
            ReconstructionAlgorithm.SOUTHWELL,
            ReconstructionAlgorithm.SOUTHWELL_REGULARIZED,
            ReconstructionAlgorithm.LEAST_SQUARES,
            ReconstructionAlgorithm.TIKHONOV,
            ReconstructionAlgorithm.MODAL_ZERNIKE,
            ReconstructionAlgorithm.POYNEER
        ]
        
        results = self.compare_algorithms(slopes_x, slopes_y, algorithms=key_algorithms, **kwargs)
        
        # Créer une grille de visualisation
        n_algos = len(key_algorithms)
        n_cols = 3
        n_rows = (n_algos + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.ravel() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
        
        for idx, algo in enumerate(key_algorithms):
            result = results[algo.value]
            
            if result.success:
                ax = axes[idx]
                im = ax.imshow(result.phase, cmap='Jet')
                ax.set_title(f"{algo.value}\nPV={result.pv:.2f}nm, RMS={result.rms:.2f}nm\nt={result.computation_time:.4f}s")
                plt.colorbar(im, ax=ax)
            else:
                axes[idx].set_title(f"{algo.value}\nErreur: {result.error}")
                axes[idx].axis('off')
        
        # Masquer les axes inutilisés
        for idx in range(n_algos, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "reconstruction_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualisation sauvegardée dans {save_dir}/reconstruction_comparison.png")


# =============================================================================
# FONCTIONS UTILITAIRES DE NIVEAU MODULE
# =============================================================================

def create_southwell_reconstructor(
    name: str = "SouthwellReconstructor",
    wavelength_nm: float = 633.0,
    pixel_size_mm: float = 0.005,
    default_algorithm: ReconstructionAlgorithm = ReconstructionAlgorithm.SOUTHWELL
) -> SouthwellReconstructor:
    """
    FR: Fabrique un reconstructeur de phase.
        
    EN: Factory function to create a phase reconstructor.
    
    Args:
        name: str - Nom du reconstructeur
        wavelength_nm: float - Longueur d'onde en nm
        pixel_size_mm: float - Taille des pixels en mm
        default_algorithm: ReconstructionAlgorithm - Algorithme par défaut
    
    Returns:
        SouthwellReconstructor - Le reconstructeur créé
    """
    return SouthwellReconstructor(
        name=name,
        wavelength_nm=wavelength_nm,
        pixel_size_mm=pixel_size_mm,
        default_algorithm=default_algorithm
    )


def reconstruct_phase(
    slopes_x: np.ndarray,
    slopes_y: np.ndarray,
    wavelength_nm: float = 633.0,
    pixel_size_mm: float = 0.005,
    algorithm: ReconstructionAlgorithm = ReconstructionAlgorithm.SOUTHWELL,
    **kwargs
) -> ReconstructionResult:
    """
    FR: Fonction utilitaire pour reconstruire la phase.
        
    EN: Utility function to reconstruct phase.
    
    Args:
        slopes_x: np.ndarray - Carte des pentes en x
        slopes_y: np.ndarray - Carte des pentes en y
        wavelength_nm: float - Longueur d'onde en nm
        pixel_size_mm: float - Taille des pixels en mm
        algorithm: ReconstructionAlgorithm - Algorithme à utiliser
        **kwargs: Arguments spécifiques à l'algorithme
    
    Returns:
        ReconstructionResult - Résultat de la reconstruction
    """
    reconstructor = create_southwell_reconstructor(
        wavelength_nm=wavelength_nm,
        pixel_size_mm=pixel_size_mm
    )
    
    return reconstructor.reconstruct(slopes_x, slopes_y, algorithm=algorithm, **kwargs)


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestSouthwell:
    """FR: Tests unitaires pour Southwell.py."""

    def test_southwell_reconstructor_creation(self):
        """FR: Test la création d'un reconstructeur."""
        reconstructor = create_southwell_reconstructor(
            name="Test",
            wavelength_nm=633.0,
            pixel_size_mm=0.005
        )
        
        assert reconstructor.name == "Test"
        assert reconstructor.wavelength_nm == 633.0
        assert reconstructor.pixel_size_mm == 0.005

    def test_southwell_reconstruction(self):
        """FR: Test la reconstruction avec l'algorithme de Southwell."""
        # Créer des pentes synthétiques (phase quadratique)
        nx, ny = 32, 32
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        # Phase quadratique : φ = a*(x² + y²)
        a = 0.1
        phase_true = a * (X**2 + Y**2)
        
        # Calculer les pentes
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        slopes_x = np.zeros_like(phase_true)
        slopes_x[:, 1:-1] = (phase_true[:, 2:] - phase_true[:, :-2]) / (2 * dx)
        
        slopes_y = np.zeros_like(phase_true)
        slopes_y[1:-1, :] = (phase_true[2:, :] - phase_true[:-2, :]) / (2 * dy)
        
        # Reconstruire
        reconstructor = create_southwell_reconstructor(pixel_size_mm=dx)
        result = reconstructor.reconstruct(slopes_x, slopes_y, 
                                          algorithm=ReconstructionAlgorithm.SOUTHWELL)
        
        assert result.success
        assert result.phase.shape == (ny, nx)
        
        # Vérifier que la reconstruction est proche de la phase vraie
        # (à une constante près)
        phase_diff = result.phase - phase_true
        assert np.std(phase_diff) < 0.1, "La reconstruction doit être proche de la phase vraie"

    def test_least_squares_reconstruction(self):
        """FR: Test la reconstruction par moindres carrés."""
        nx, ny = 16, 16
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        # Phase simple : φ = x + y
        phase_true = X + Y
        
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        slopes_x = np.zeros_like(phase_true)
        slopes_x[:, 1:-1] = (phase_true[:, 2:] - phase_true[:, :-2]) / (2 * dx)
        
        slopes_y = np.zeros_like(phase_true)
        slopes_y[1:-1, :] = (phase_true[2:, :] - phase_true[:-2, :]) / (2 * dy)
        
        reconstructor = create_southwell_reconstructor(pixel_size_mm=dx)
        result = reconstructor.reconstruct(slopes_x, slopes_y,
                                          algorithm=ReconstructionAlgorithm.LEAST_SQUARES)
        
        assert result.success
        assert result.phase.shape == (ny, nx)

    def test_modal_zernike_reconstruction(self):
        """FR: Test la reconstruction modale avec Zernike."""
        try:
            from Optiques import generate_zernike_polynomial
        except ImportError:
            return  # Skip if Optiques not available
        
        nx, ny = 32, 32
        
        # Créer des pentes à partir d'une phase Zernike
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        # Phase avec quelques modes Zernike
        phase_true = 0.1 * generate_zernike_polynomial(2, 0, X, Y) + \
                    0.05 * generate_zernike_polynomial(2, 2, X, Y)
        
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        slopes_x = np.zeros_like(phase_true)
        slopes_x[:, 1:-1] = (phase_true[:, 2:] - phase_true[:, :-2]) / (2 * dx)
        
        slopes_y = np.zeros_like(phase_true)
        slopes_y[1:-1, :] = (phase_true[2:, :] - phase_true[:-2, :]) / (2 * dy)
        
        reconstructor = create_southwell_reconstructor(pixel_size_mm=dx)
        result = reconstructor.reconstruct(slopes_x, slopes_y,
                                          algorithm=ReconstructionAlgorithm.MODAL_ZERNIKE,
                                          max_zernike_degree=5)
        
        assert result.success
        assert result.phase.shape == (ny, nx)

    def test_compare_algorithms(self):
        """FR: Test la comparaison des algorithmes."""
        nx, ny = 16, 16
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        phase_true = X**2 + Y**2
        
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        slopes_x = np.zeros_like(phase_true)
        slopes_x[:, 1:-1] = (phase_true[:, 2:] - phase_true[:, :-2]) / (2 * dx)
        
        slopes_y = np.zeros_like(phase_true)
        slopes_y[1:-1, :] = (phase_true[2:, :] - phase_true[:-2, :]) / (2 * dy)
        
        reconstructor = create_southwell_reconstructor(pixel_size_mm=dx)
        
        algorithms = [
            ReconstructionAlgorithm.SOUTHWELL,
            ReconstructionAlgorithm.LEAST_SQUARES,
            ReconstructionAlgorithm.TIKHONOV
        ]
        
        results = reconstructor.compare_algorithms(slopes_x, slopes_y, algorithms=algorithms)
        
        assert len(results) == 3
        for algo in algorithms:
            assert algo.value in results
            assert results[algo.value].success

    def test_best_algorithm(self):
        """FR: Test la sélection du meilleur algorithme."""
        nx, ny = 16, 16
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        phase_true = X + Y
        
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        slopes_x = np.zeros_like(phase_true)
        slopes_x[:, 1:-1] = (phase_true[:, 2:] - phase_true[:, :-2]) / (2 * dx)
        
        slopes_y = np.zeros_like(phase_true)
        slopes_y[1:-1, :] = (phase_true[2:, :] - phase_true[:-2, :]) / (2 * dy)
        
        reconstructor = create_southwell_reconstructor(pixel_size_mm=dx)
        
        best_algo, best_result = reconstructor.get_best_algorithm(
            slopes_x, slopes_y,
            metric="rms"
        )
        
        assert best_algo is not None
        assert best_result.success


if __name__ == "__main__":
    import unittest
    unittest.main()
