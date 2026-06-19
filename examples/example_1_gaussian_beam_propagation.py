"""
Example 1: Gaussian Beam Propagation
FR: Exemple de génération et propagation d'un faisceau gaussien.
    Démonstration de l'intégration entre Beam.py, Propagation.py et Visualization.py.
    Teste différentes tailles de grille et normalisations.

EN: Example of Gaussian beam generation and propagation.
    Demonstrates integration between Beam.py, Propagation.py and Visualization.py.
    Tests different grid sizes and normalizations.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Mode non-interactif pour éviter les erreurs
import matplotlib.pyplot as plt
from Beam import Beam
from Propagation import Propagation
from Visualization import plot_intensity, plot_phase, plot_beam_map


def run_gaussian_beam_example():
    """FR: Exécute l'exemple de propagation d'un faisceau gaussien."""
    print("\n" + "="*80)
    print("Example 1: Gaussian Beam Propagation")
    print("="*80)
    
    # =========================================================================
    # 1. Génération du faisceau initial
    # =========================================================================
    print("\n--- Génération du faisceau gaussien ---")
    
    # Paramètres
    wavelength_nm = 633.0  # Longueur d'onde en nm (rouge)
    diameter_mm = 10.0     # Diamètre du faisceau en mm
    energy = 1.0           # Énergie en a.u.
    sigma_mm = 2.0        # Taille du waist en mm
    
    # Créer le faisceau
    beam = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=diameter_mm,
        energy=energy,
        energy_unit="a.u.",
        num_points=256,      # Taille de la grille
        coherence="coherent"
    )
    
    # Générer le champ électrique gaussien
    electric_field = beam.generate_electric_field(
        method="gaussian",
        sigma_mm=sigma_mm
    )
    
    # Extraire l'intensité et la phase
    intensity = beam.compute_intensity_from_electric_field(electric_field)
    phase = beam.extract_phase_from_electric_field(electric_field)
    
    # Calculer PV et RMS
    pv, rms = beam.compute_pv_rms(phase)
    print(f"Faisceau initial: PV={pv:.2f} nm, RMS={rms:.2f} nm")
    print(f"Énergie totale: {np.sum(intensity):.4f} (a.u.)")
    
    # Visualisation du faisceau initial
    plot_intensity(intensity, diameter_mm, intensity_unit="a.u.", title="Gaussian Beam - Intensity (Initial)")
    plot_phase(phase, diameter_mm, title="Gaussian Beam - Phase (Initial)")
    plt.savefig('examples/output/example1_initial_beam.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 2. Propagation avec différentes méthodes
    # =========================================================================
    print("\n--- Propagation du faisceau ---")
    
    # Distance de propagation
    propagation_distance_mm = 500.0  # 50 cm
    
    # Méthodes de propagation à tester
    methods = ["angular_spectrum", "fraunhofer", "fresnel"]
    
    for method in methods:
        print(f"\n--- Méthode: {method} ---")
        
        # Créer le propagateur
        propagator = Propagation(
            wavelength_nm=wavelength_nm,
            propagation_distance_mm=propagation_distance_mm,
            input_diameter_mm=diameter_mm,
            output_diameter_mm=diameter_mm,
            num_points=256,
            coherence="coherent",
            method=method
        )
        
        # Propage le champ électrique
        if method in ["hermite_gauss", "laguerre_gauss"]:
            propagated_field, metrics = propagator.propagate(electric_field, max_modes=10)
            print(f"  Erreur RMS intensité: {metrics['intensity_error_rms']:.4f}")
            print(f"  Erreur RMS phase: {metrics['phase_error_rms_nm']:.2f} nm")
        else:
            propagated_field = propagator.propagate(electric_field)
        
        # Extraire l'intensité et la phase propagées
        propagated_intensity = beam.compute_intensity_from_electric_field(propagated_field)
        propagated_phase = beam.extract_phase_from_electric_field(propagated_field)
        
        # Calculer PV et RMS
        pv_prop, rms_prop = beam.compute_pv_rms(propagated_phase)
        print(f"  Faisceau propagé: PV={pv_prop:.2f} nm, RMS={rms_prop:.2f} nm")
        print(f"  Énergie propagée: {np.sum(propagated_intensity):.4f} (a.u.)")
        
        # Visualisation
        plot_intensity(
            propagated_intensity, 
            diameter_mm, 
            intensity_unit="a.u.", 
            title=f"Gaussian Beam - Intensity ({method}, z={propagation_distance_mm}mm)"
        )
        plot_phase(
            propagated_phase, 
            diameter_mm, 
            title=f"Gaussian Beam - Phase ({method}, z={propagation_distance_mm}mm)"
        )
        plt.savefig(f'examples/output/example1_propagated_{method}.png', dpi=150, bbox_inches='tight')
        plt.close('all')
    
    # =========================================================================
    # 3. Test avec différentes tailles de grille
    # =========================================================================
    print("\n--- Test avec différentes tailles de grille ---")
    
    grid_sizes = [64, 128, 256, 512]
    
    for size in grid_sizes:
        print(f"\nTaille de grille: {size}x{size}")
        
        # Créer le faisceau
        beam_test = Beam(
            wavelength_nm=wavelength_nm,
            diameter_mm=diameter_mm,
            energy=energy,
            num_points=size
        )
        
        # Générer le champ électrique
        electric_field_test = beam_test.generate_electric_field(method="gaussian", sigma_mm=sigma_mm)
        
        # Propager
        propagator_test = Propagation(
            wavelength_nm=wavelength_nm,
            propagation_distance_mm=propagation_distance_mm,
            input_diameter_mm=diameter_mm,
            num_points=size,
            method="angular_spectrum"
        )
        propagated_field_test = propagator_test.propagate(electric_field_test)
        
        # Calculer le temps (approximatif)
        print(f"  Temps estimé: O({size}^2 log {size}) opérations")
    
    # =============================================================================
    # 4. Test avec différentes normalisations
    # =========================================================================
    print("\n--- Test avec différentes normalisations ---")
    
    normalizations = ["RMS", "PV"]
    
    for norm in normalizations:
        print(f"\nNormalisation: {norm}")
        
        # Créer le faisceau
        beam_norm = Beam(
            wavelength_nm=wavelength_nm,
            diameter_mm=diameter_mm,
            energy=energy,
            num_points=128
        )
        
        # Générer une phase avec normalisation spécifique
        phase_norm = beam_norm.generate_phase(
            method="random_zernike",
            n_modes=5,
            max_amplitude_nm=100.0,
            normalization=norm
        )
        
        # Générer le champ électrique à partir de l'intensité et de la phase
        electric_field_norm = beam_norm.generate_electric_field(
            intensity=beam_norm.generate_intensity(method="gaussian", sigma_mm=sigma_mm),
            phase=phase_norm,
            method="from_intensity_phase"
        )
        
        # Propager
        propagator_norm = Propagation(
            wavelength_nm=wavelength_nm,
            propagation_distance_mm=propagation_distance_mm,
            input_diameter_mm=diameter_mm,
            num_points=128,
            method="angular_spectrum"
        )
        propagated_field_norm = propagator_norm.propagate(electric_field_norm)
        
        # Extraire la phase propagée
        propagated_phase_norm = beam_norm.extract_phase_from_electric_field(propagated_field_norm)
        pv_norm, rms_norm = beam_norm.compute_pv_rms(propagated_phase_norm)
        print(f"  Phase propagée: PV={pv_norm:.2f} nm, RMS={rms_norm:.2f} nm")
    
    print("\n" + "="*80)
    print("Example 1 terminé avec succès !")
    print("Les images ont été sauvegardées dans examples/output/")
    print("="*80)


if __name__ == "__main__":
    # Créer le répertoire de sortie
    import os
    os.makedirs('examples/output', exist_ok=True)
    run_gaussian_beam_example()
