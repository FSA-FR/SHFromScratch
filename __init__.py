"""
SHFromScratch
FR: Package Python pour la simulation de capteurs Shack-Hartmann.
    Ce package permet de simuler la propagation d'un faisceau à travers une matrice de microlentilles ou de microtrous
    jusqu'au plan d'une caméra virtuelle, puis le calcul de pente locale et la reconstruction de la phase.

EN: Python package for Shack-Hartmann sensor simulation.
    This package allows simulating the propagation of a beam through a microlens or microhole array
    up to a virtual camera plane, then calculating local slopes and phase reconstruction.

Structure du package/Package structure:
- Beam.py: Génération et gestion des faisceaux optiques.
- MathAndPhysicsTools.py: Fonctions mathématiques et physiques réutilisables.
- Propagation.py: Propagation analytique ou numérique (FFT) des faisceaux.
- Material_Behaviour.py: Gestion du chromatisme et des comportements thermiques.
- Microstructure.py: Génération de matrices de microlentilles ou microtrous.
- Optiques.py: Génération d'optiques (lentilles, beamsplitters, etc.).
- Camera.py: Création de capteurs virtuels (parfaits ou réels).
- Shack_Hartmann.py: Calcul des centroïdes et des pentes locales.
- Southwell.py: Algorithmes de reconstruction de phase.
- Visualization.py: Fonctions d'affichage.
- Examples.py: Exemples d'utilisation.
- Simulation.py: Simulation complète du Shack-Hartmann.

Author: FSA-FR
Version: 0.1.0
Repository: https://github.com/FSA-FR/SHFromScratch
"""

__version__ = "0.1.0"
__author__ = "FSA-FR"

# Liste des modules à importer pour une utilisation directe
__all__ = [
    "Beam",
    "Propagation",
    "Material_Behaviour",
    "Microstructure",
    "Optiques",
    "Camera",
    "Shack_Hartmann",
    "Southwell",
    "Visualization",
    "MathAndPhysicsTools",
    "Examples",
    "Simulation",
]

# Import des modules principaux (seront ajoutés au fur et à mesure)
from . import Beam
from . import MathAndPhysicsTools