"""
Example 2: Zernike Beam Propagation
FR: Exemple de génération et propagation d'un faisceau avec phase Zernike.
    Démonstration de la génération de phase avec des polynômes de Zernike,
    puis propagation et visualisation des résultats.

EN: Example of Zernike beam generation and propagation.
    Demonstrates phase generation with Zernike polynomials,
    then propagation and visualization of results.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Beam import Beam
from Propagation import Propagation
from Visualization import plot_intensity, plot_phase, plot_beam_map


def run_zernike_beam_example():
    """FR: Exécute l'exemple de propagation d'un faisceau avec phase Zernike."""
    print("\n" + "="*80)
    print("Example 2: Zernike Beam Propagation")
    print("="*80)
    
    # =========================================================================
    # 1. Génération du faisceau avec phase Zernike
    # =========================================================================
    print("\n--- Génération du faisceau avec phase Zernike ---")
    
    # Paramètres
    wavelength_nm = 633.0
    diameter_mm = 10.0
    energy = 1.0
    num_points = 256
    
    # Créer le faisceau
    beam = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=diameter_mm,
        energy=energy,
        num_points=num_points,
        coherence="coherent"
    )
    
    # Générer une intensité gaussienne
    intensity = beam.generate_intensity(method="gaussian", sigma_mm=2.0)
    
    # Générer une phase avec des polynômes de Zernike
    print("\nTest avec différentes ordinations et nombres de modes:")
    
    ordinations = ["Noll", "Wyant"]
    mode_counts = [5, 10, 20]
    
    for ordination in ordinations:
        for n_modes in mode_counts:
            print(f"\n  Ordination: {ordination}, Nombre de modes: {n_modes}")
            
            # Générer la phase
            phase = beam.generate_phase(
                method="random_zernike",
                n_modes=n_modes,
                ordination=ordination,
                max_amplitude_nm=100.0,
                normalization="RMS"
            )
            
            # Calculer PV et RMS
            pv, rms = beam.compute_pv_rms(phase)
            print(f"    Phase: PV={pv:.2f} nm, RMS={rms:.2f} nm")
            
            # Générer le champ électrique
            electric_field = beam.generate_electric_field(
                intensity=intensity,
                phase=phase,
                method="from_intensity_phase"
            )
            
            # Visualisation
            plot_phase(phase, diameter_mm, title=f"Phase Zernike ({ordination}, {n_modes} modes)")
            plt.savefig(f'examples/output/example2_zernike_{ordination}_{n_modes}_modes.png', dpi=150, bbox_inches='tight')
            plt.close('all')
    
    # =========================================================================
    # 2. Propagation du faisceau avec phase Zernike
    # =========================================================================
    print("\n--- Propagation du faisceau avec phase Zernike ---")
    
    # Utiliser une configuration spécifique
    ordination = "Noll"
    n_modes = 10
    propagation_distance_mm = 500.0
    
    # Générer la phase Zernike
    phase = beam.generate_phase(
        method="random_zernike",
        n_modes=n_modes,
        ordination=ordination,
        max_amplitude_nm=100.0,
        normalization="RMS"
    )
    
    # Générer le champ électrique
    electric_field = beam.generate_electric_field(
        intensity=intensity,
        phase=phase,
        method="from_intensity_phase"
    )
    
    # Propager avec différentes méthodes
    methods = ["angular_spectrum", "fraunhofer"]
    
    for method in methods:
        print(f"\nMéthode: {method}")
        
        propagator = Propagation(
            wavelength_nm=wavelength_nm,
            propagation_distance_mm=propagation_distance_mm,
            input_diameter_mm=diameter_mm,
            num_points=num_points,
            method=method
        )
        
        propagated_field = propagator.propagate(electric_field)
        
        # Extraire l'intensité et la phase propagées
        propagated_intensity = beam.compute_intensity_from_electric_field(propagated_field)
        propagated_phase = beam.extract_phase_from_electric_field(propagated_field)
        
        pv_prop, rms_prop = beam.compute_pv_rms(propagated_phase)
        print(f"  Phase propagée: PV={pv_prop:.2f} nm, RMS={rms_prop:.2f} nm")
        
        # Visualisation
        plot_intensity(
            propagated_intensity,
            diameter_mm,
            title=f"Intensity ({method}, Zernike {ordination}, {n_modes} modes)"
        )
        plot_phase(
            propagated_phase,
            diameter_mm,
            title=f"Phase ({method}, Zernike {ordination}, {n_modes} modes)"
        )
        plt.savefig(f'examples/output/example2_propagated_{method}_zernike.png', dpi=150, bbox_inches='tight')
        plt.close('all')
    
    # =========================================================================
    # 3. Comparaison des normalisations
    # =========================================================================
    print("\n--- Comparaison des normalisations (PV vs RMS) ---")
    
    for norm in ["PV", "RMS"]:
        print(f"\nNormalisation: {norm}")
        
        phase_norm = beam.generate_phase(
            method="random_zernike",
            n_modes=10,
            ordination="Noll",
            max_amplitude_nm=100.0,
            normalization=norm
        )
        
        pv_norm, rms_norm = beam.compute_pv_rms(phase_norm)
        print(f"  Phase: PV={pv_norm:.2f} nm, RMS={rms_norm:.2f} nm")
        
        # Vérification de la normalisation
        if norm == "PV":
            assert abs(pv_norm - 100.0) < 1.0, f"Normalisation PV échouée: attendu 100 nm, obtenu {pv_norm:.2f} nm"
        elif norm == "RMS":
            assert abs(rms_norm - 100.0) < 1.0, f"Normalisation RMS échouée: attendu 100 nm, obtenu {rms_norm:.2f} nm"
        
        print(f"  ✅ Normalisation {norm} validée")
    
    print("\n" + "="*80)
    print("Example 2 terminé avec succès !")
    print("Les images ont été sauvegardées dans examples/output/")
    print("="*80)


if __name__ == "__main__":
    import os
    os.makedirs('examples/output', exist_ok=True)
    run_zernike_beam_example()
