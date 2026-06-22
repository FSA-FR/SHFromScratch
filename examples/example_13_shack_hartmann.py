"""
Example 13: Shack-Hartmann Wavefront Sensor
FR: Exemple complet d'utilisation du module Shack_Hartmann.py.
    Démonstration de :
    - Création d'un système Shack-Hartmann complet (matrice de microlentilles + caméra)
    - Simulation avec un faisceau gaussien
    - Calcul des centroïdes avec 4 algorithmes différents
    - Calcul des pentes locales de phase (dφ/dx, dφ/dy)
    - Estimation des erreurs sur les centroïdes et les pentes
    - Visualisation :
        * Carte des tâches d'Airy (spots)
        * Carte des centroïdes (marqués sur l'image des spots)
        * Cartes des pentes locales (X et Y)
        * Cartes d'erreur (centroïdes et pentes)
    - Simulation avec un faisceau aberré (Zernike)
    - Comparaison des algorithmes de centroïdes
    - Export des données

    Unités :
    - Longueurs : mm (positions), µm (microlentilles)
    - Longueur d'onde : nm
    - Phase : nm, rad, mrad
    - Pentes : rad/mm, mrad/mm

EN: Complete example of using Shack_Hartmann.py module.
    Demonstrates:
    - Creation of a complete Shack-Hartmann system (microlens array + camera)
    - Simulation with a Gaussian beam
    - Centroid calculation with 4 different algorithms
    - Local phase slope calculation (dφ/dx, dφ/dy)
    - Error estimation on centroids and slopes
    - Visualization:
        * Airy spot map
        * Centroid map (marked on spot image)
        * Local slope maps (X and Y)
        * Error maps (centroids and slopes)
    - Simulation with an aberrated beam (Zernike)
    - Comparison of centroid algorithms
    - Data export

    Units:
    - Lengths: mm (positions), µm (microlenses)
    - Wavelength: nm
    - Phase: nm, rad, mrad
    - Slopes: rad/mm, mrad/mm

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - matplotlib
    - Shack_Hartmann.py
    - Beam.py
    - Propagation.py
    - Microstructure.py
    - Camera.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

from Shack_Hartmann import (
    ShackHartmann, create_shack_hartmann,
    CentroidAlgorithm, SlopeCalculationMethod
)
from Beam import Beam
from Optiques import WaveFrontError
from Microstructure import create_microlens_array, ArrayPattern
from Camera import RealCamera


def run_shack_hartmann_example():
    """
    FR: Exécute l'exemple Shack-Hartmann.
        Démonstration complète des fonctionnalités du module.
    """
    print("\n" + "="*80)
    print("Example 13: Shack-Hartmann Wavefront Sensor")
    print("="*80)
    
    # Paramètres globaux
    wavelength_nm = 633.0
    output_dir = 'examples/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # 1. CRÉATION D'UN SYSTÈME SHACK-HARTMANN PAR DÉFAUT
    # =========================================================================
    print("\n--- 1. Création d'un système Shack-Hartmann par défaut ---")
    
    print("\n1.1 Création avec la fabrique")
    sh_default = create_shack_hartmann(
        name="ShackHartmann_Default",
        num_microlenses_x=8,
        num_microlenses_y=8,
        microlens_pitch_mm=0.5,  # Espacement bord-à-bord
        focal_length_mm=20.0,
        camera_num_pixels_x=512,
        camera_num_pixels_y=512,
        camera_pixel_size_um=5.0,  # 5 µm
        wavelength_nm=wavelength_nm,
        centroid_algorithm=CentroidAlgorithm.WEIGHTED_CENTROID,
        slope_method=SlopeCalculationMethod.FINITE_DIFFERENCE,
        display=True,
        display_dir=output_dir
    )
    
    print(f"  {sh_default}")
    print(f"  Matrice de microlentilles: {sh_default.microlens_array}")
    print(f"  Caméra: {sh_default.camera}")
    print(f"  Algorithme de centroïde: {sh_default.centroid_algorithm.value}")
    print(f"  Méthode de pente: {sh_default.slope_method.value}")
    
    # =========================================================================
    # 2. SIMULATION AVEC UN FAISCEAU GAUSSIEN
    # =========================================================================
    print("\n--- 2. Simulation avec un faisceau gaussien ---")
    
    # 2.1 Créer un faisceau gaussien
    print("\n2.1 Création d'un faisceau gaussien")
    beam_gaussian = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        num_points=512
    )
    electric_field_gaussian = beam_gaussian.generate_electric_field(
        method="gaussian",
        sigma_mm=1.0
    )
    beam_gaussian.electric_field = electric_field_gaussian
    beam_gaussian.intensity = beam_gaussian.compute_intensity_from_electric_field(electric_field_gaussian)
    beam_gaussian.phase = beam_gaussian.extract_phase_from_electric_field(electric_field_gaussian)
    
    # Calculer PV et RMS du faisceau initial
    pv_initial, rms_initial = beam_gaussian.compute_pv_rms(beam_gaussian.intensity)
    print(f"  Faisceau initial: PV={pv_initial:.2f}, RMS={rms_initial:.2f}")
    
    # 2.2 Simuler le système Shack-Hartmann
    print("\n2.2 Simulation Shack-Hartmann")
    sh_default.simulate(beam_gaussian)
    
    # Afficher les résultats
    print(f"  Tâches détectées: {len(sh_default.centroids)} spots")
    print(f"  Forme des cartes de pentes: {sh_default.slopes_x.shape}")
    
    # 2.3 Statistiques des résultats
    print("\n2.3 Statistiques des résultats")
    stats = sh_default.get_slope_statistics()
    
    print(f"  Pentes X:")
    print(f"    PV: {stats['slopes_x']['pv']:.6f} rad/mm")
    print(f"    RMS: {stats['slopes_x']['rms']:.6f} rad/mm")
    print(f"    Moyenne: {stats['slopes_x']['mean']:.6f} rad/mm")
    
    print(f"  Pentes Y:")
    print(f"    PV: {stats['slopes_y']['pv']:.6f} rad/mm")
    print(f"    RMS: {stats['slopes_y']['rms']:.6f} rad/mm")
    print(f"    Moyenne: {stats['slopes_y']['mean']:.6f} rad/mm")
    
    print(f"  Erreurs sur les pentes X:")
    print(f"    Moyenne: {stats['slope_errors_x']['mean']:.6f} rad/mm")
    print(f"    RMS: {stats['slope_errors_x']['rms']:.6f} rad/mm")
    
    print(f"  Erreurs sur les pentes Y:")
    print(f"    Moyenne: {stats['slope_errors_y']['mean']:.6f} rad/mm")
    print(f"    RMS: {stats['slope_errors_y']['rms']:.6f} rad/mm")
    
    # 2.4 Visualiser toutes les cartes
    print("\n2.4 Visualisation de toutes les cartes")
    sh_default.visualize_all(output_dir=output_dir)
    
    # =========================================================================
    # 3. COMPARAISON DES ALGORITHMES DE CENTROÏDES
    # =========================================================================
    print("\n--- 3. Comparaison des algorithmes de centroïdes ---")
    
    algorithms = [
        (CentroidAlgorithm.WEIGHTED_CENTROID, "Barycentre pondéré"),
        (CentroidAlgorithm.THRESHOLDED_CENTROID, "Barycentre avec seuil"),
        (CentroidAlgorithm.GAUSSIAN_FIT, "Ajustement gaussien"),
        (CentroidAlgorithm.MOMENT_BASED, "Basé sur les moments")
    ]
    
    results = {}
    
    for algo, algo_name in algorithms:
        print(f"\n3.{algorithms.index((algo, algo_name))+1} {algo_name}")
        
        # Créer un système avec cet algorithme
        sh_algo = create_shack_hartmann(
            name=f"ShackHartmann_{algo.value}",
            num_microlenses_x=5,
            num_microlenses_y=5,
            focal_length_mm=20.0,
            centroid_algorithm=algo,
            display=False
        )
        
        # Simuler avec le même faisceau
        beam_algo = Beam(
            wavelength_nm=wavelength_nm,
            diameter_mm=5.0,
            num_points=256
        )
        electric_field_algo = beam_algo.generate_electric_field(
            method="gaussian",
            sigma_mm=1.0
        )
        beam_algo.electric_field = electric_field_algo
        beam_algo.intensity = beam_algo.compute_intensity_from_electric_field(electric_field_algo)
        
        sh_algo.simulate(beam_algo)
        
        # Calculer les statistiques
        stats_algo = sh_algo.get_slope_statistics()
        centroid_mean_error = np.mean(sh_algo.centroid_errors)
        
        results[algo.value] = {
            'centroid_mean_error': centroid_mean_error,
            'slope_x_pv': stats_algo['slopes_x']['pv'],
            'slope_x_rms': stats_algo['slopes_x']['rms'],
            'slope_y_pv': stats_algo['slopes_y']['pv'],
            'slope_y_rms': stats_algo['slopes_y']['rms'],
            'slope_error_x_mean': stats_algo['slope_errors_x']['mean'],
            'slope_error_y_mean': stats_algo['slope_errors_y']['mean']
        }
        
        print(f"  Erreur moyenne sur les centroïdes: {centroid_mean_error:.4f} pixels")
        print(f"  Pentes X: PV={stats_algo['slopes_x']['pv']:.6f}, RMS={stats_algo['slopes_x']['rms']:.6f} rad/mm")
        print(f"  Pentes Y: PV={stats_algo['slopes_y']['pv']:.6f}, RMS={stats_algo['slopes_y']['rms']:.6f} rad/mm")
        
        # Visualiser les centroïdes
        sh_algo.visualize_centroids(
            save_path=os.path.join(output_dir, f"centroids_{algo.value}.png"),
            show=False
        )
    
    # 3.5 Comparaison visuelle des algorithmes
    print("\n3.5 Comparaison visuelle des algorithmes")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    titles = [
        "Barycentre pondéré",
        "Barycentre avec seuil",
        "Ajustement gaussien",
        "Basé sur les moments"
    ]
    
    for idx, (algo, _) in enumerate(algorithms):
        sh_algo = create_shack_hartmann(
            name=f"Compare_{algo.value}",
            num_microlenses_x=5,
            num_microlenses_y=5,
            focal_length_mm=20.0,
            centroid_algorithm=algo,
            display=False
        )
        
        beam_compare = Beam(
            wavelength_nm=wavelength_nm,
            diameter_mm=5.0,
            num_points=256
        )
        electric_field_compare = beam_compare.generate_electric_field(
            method="gaussian",
            sigma_mm=1.0
        )
        beam_compare.electric_field = electric_field_compare
        beam_compare.intensity = beam_compare.compute_intensity_from_electric_field(electric_field_compare)
        
        sh_algo.simulate(beam_compare)
        centroid_map = sh_algo.get_centroid_map()
        
        row, col = idx // 2, idx % 2
        axes[row, col].imshow(centroid_map, cmap='hot')
        axes[row, col].set_title(titles[idx])
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "centroid_algorithms_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Image sauvegardée: centroid_algorithms_comparison.png")
    
    # 3.6 Tableau comparatif
    print("\n3.6 Tableau comparatif des algorithmes")
    print("\n{:<25} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
        "Algorithme",
        "Erreur centroïde",
        "PV X (rad/mm)",
        "PV Y (rad/mm)",
        "Erreur pente X",
        "Erreur pente Y"
    ))
    print("-" * 110)
    
    for algo, algo_name in algorithms:
        data = results[algo.value]
        print("{:<25} {:<15.4f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            algo_name,
            data['centroid_mean_error'],
            data['slope_x_pv'],
            data['slope_y_pv'],
            data['slope_error_x_mean'],
            data['slope_error_y_mean']
        ))
    
    # =========================================================================
    # 4. SIMULATION AVEC UN FAISCEAU ABERRÉ (ZERNIKE)
    # =========================================================================
    print("\n--- 4. Simulation avec un faisceau aberré ---")
    
    # 4.1 Créer un faisceau avec des aberrations
    print("\n4.1 Faisceau avec aberrations (Zernike)")
    beam_aberrated = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        num_points=512
    )
    electric_field_aberrated = beam_aberrated.generate_electric_field(
        method="gaussian",
        sigma_mm=1.0
    )
    
    # Ajouter des aberrations (Zernike)
    wfe_aberrated = WaveFrontError(
        surface_roughness_nm=1.0,
        parallelism_arcsec=0.5,
        zernike_coefficients={
            (2, 0): 10.0,   # Defocus (Z4)
            (2, 2): 5.0,    # Astigmatisme (Z5)
            (3, 1): 3.0,    # Coma (Z7)
            (3, 3): 2.0     # Coma secondaire (Z8)
        },
        seed=42
    )
    
    # Appliquer les aberrations au faisceau
    x = np.linspace(-beam_aberrated.diameter_mm/2, beam_aberrated.diameter_mm/2, beam_aberrated.num_points)
    y = np.linspace(-beam_aberrated.diameter_mm/2, beam_aberrated.diameter_mm/2, beam_aberrated.num_points)
    X, Y = np.meshgrid(x, y)
    
    phase_map_nm = wfe_aberrated.generate_phase_map(X, Y, beam_aberrated.wavelength_nm)
    phase_rad = phase_map_nm * 2 * np.pi / beam_aberrated.wavelength_nm
    
    aberrated_field = np.abs(electric_field_aberrated) * np.exp(
        1j * (np.angle(electric_field_aberrated) + phase_rad)
    )
    beam_aberrated.electric_field = aberrated_field
    beam_aberrated.intensity = beam_aberrated.compute_intensity_from_electric_field(aberrated_field)
    beam_aberrated.phase = beam_aberrated.extract_phase_from_electric_field(aberrated_field)
    
    # Calculer PV et RMS du faisceau aberré
    pv_aberrated, rms_aberrated = beam_aberrated.compute_pv_rms(beam_aberrated.phase)
    print(f"  Faisceau aberré: PV={pv_aberrated:.2f} nm, RMS={rms_aberrated:.2f} nm")
    print(f"  Aberrations appliquées: {wfe_aberrated.zernike_coefficients}")
    
    # 4.2 Simuler avec le faisceau aberré
    print("\n4.2 Simulation Shack-Hartmann avec faisceau aberré")
    sh_aberrated = create_shack_hartmann(
        name="ShackHartmann_Aberrated",
        num_microlenses_x=8,
        num_microlenses_y=8,
        focal_length_mm=20.0,
        display=True,
        display_dir=output_dir
    )
    
    sh_aberrated.simulate(beam_aberrated)
    
    # 4.3 Statistiques des résultats
    print("\n4.3 Statistiques des résultats (faisceau aberré)")
    stats_aberrated = sh_aberrated.get_slope_statistics()
    
    print(f"  Pentes X:")
    print(f"    PV: {stats_aberrated['slopes_x']['pv']:.6f} rad/mm")
    print(f"    RMS: {stats_aberrated['slopes_x']['rms']:.6f} rad/mm")
    print(f"    Moyenne: {stats_aberrated['slopes_x']['mean']:.6f} rad/mm")
    
    print(f"  Pentes Y:")
    print(f"    PV: {stats_aberrated['slopes_y']['pv']:.6f} rad/mm")
    print(f"    RMS: {stats_aberrated['slopes_y']['rms']:.6f} rad/mm")
    print(f"    Moyenne: {stats_aberrated['slopes_y']['mean']:.6f} rad/mm")
    
    # 4.4 Visualiser les résultats
    print("\n4.4 Visualisation des résultats")
    sh_aberrated.visualize_all(output_dir=output_dir)
    
    # 4.5 Comparaison avec le faisceau non aberré
    print("\n4.5 Comparaison avec le faisceau non aberré")
    
    stats_default = sh_default.get_slope_statistics()
    
    print(f"\n  Faisceau non aberré:")
    print(f"    Pentes X: PV={stats_default['slopes_x']['pv']:.6f}, RMS={stats_default['slopes_x']['rms']:.6f} rad/mm")
    print(f"    Pentes Y: PV={stats_default['slopes_y']['pv']:.6f}, RMS={stats_default['slopes_y']['rms']:.6f} rad/mm")
    
    print(f"\n  Faisceau aberré:")
    print(f"    Pentes X: PV={stats_aberrated['slopes_x']['pv']:.6f}, RMS={stats_aberrated['slopes_x']['rms']:.6f} rad/mm")
    print(f"    Pentes Y: PV={stats_aberrated['slopes_y']['pv']:.6f}, RMS={stats_aberrated['slopes_y']['rms']:.6f} rad/mm")
    
    print(f"\n  Ratio PV (aberré/non aberré):")
    print(f"    Pentes X: {stats_aberrated['slopes_x']['pv'] / stats_default['slopes_x']['pv']:.2f}")
    print(f"    Pentes Y: {stats_aberrated['slopes_y']['pv'] / stats_default['slopes_y']['pv']:.2f}")
    
    # Visualisation comparative
    slopes_x_default, slopes_y_default = sh_default.get_slope_maps()
    slopes_x_aberrated, slopes_y_aberrated = sh_aberrated.get_slope_maps()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Pentes X - Non aberré
    im1 = axes[0, 0].imshow(
        slopes_x_default,
        cmap='Jet',
        extent=[-2.0, 2.0, -2.0, 2.0],
        origin='lower'
    )
    axes[0, 0].set_title("Pentes X - Faisceau non aberré\n" +
                       f"PV={stats_default['slopes_x']['pv']:.6f} rad/mm, " +
                       f"RMS={stats_default['slopes_x']['rms']:.6f} rad/mm")
    axes[0, 0].set_xlabel("x (mm)")
    axes[0, 0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0, 0], label="Pente (rad/mm)")
    
    # Pentes X - Aberré
    im2 = axes[0, 1].imshow(
        slopes_x_aberrated,
        cmap='Jet',
        extent=[-2.0, 2.0, -2.0, 2.0],
        origin='lower'
    )
    axes[0, 1].set_title("Pentes X - Faisceau aberré\n" +
                       f"PV={stats_aberrated['slopes_x']['pv']:.6f} rad/mm, " +
                       f"RMS={stats_aberrated['slopes_x']['rms']:.6f} rad/mm")
    axes[0, 1].set_xlabel("x (mm)")
    axes[0, 1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[0, 1], label="Pente (rad/mm)")
    
    # Pentes Y - Non aberré
    im3 = axes[1, 0].imshow(
        slopes_y_default,
        cmap='Jet',
        extent=[-2.0, 2.0, -2.0, 2.0],
        origin='lower'
    )
    axes[1, 0].set_title("Pentes Y - Faisceau non aberré\n" +
                       f"PV={stats_default['slopes_y']['pv']:.6f} rad/mm, " +
                       f"RMS={stats_default['slopes_y']['rms']:.6f} rad/mm")
    axes[1, 0].set_xlabel("x (mm)")
    axes[1, 0].set_ylabel("y (mm)")
    plt.colorbar(im3, ax=axes[1, 0], label="Pente (rad/mm)")
    
    # Pentes Y - Aberré
    im4 = axes[1, 1].imshow(
        slopes_y_aberrated,
        cmap='Jet',
        extent=[-2.0, 2.0, -2.0, 2.0],
        origin='lower'
    )
    axes[1, 1].set_title("Pentes Y - Faisceau aberré\n" +
                       f"PV={stats_aberrated['slopes_y']['pv']:.6f} rad/mm, " +
                       f"RMS={stats_aberrated['slopes_y']['rms']:.6f} rad/mm")
    axes[1, 1].set_xlabel("x (mm)")
    axes[1, 1].set_ylabel("y (mm)")
    plt.colorbar(im4, ax=axes[1, 1], label="Pente (rad/mm)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slopes_comparison_aberrated_vs_ideal.png"), dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Image sauvegardée: slopes_comparison_aberrated_vs_ideal.png")
    
    # =========================================================================
    # 5. SIMULATION AVEC UNE MATRICE HEXAGONALE
    # =========================================================================
    print("\n--- 5. Simulation avec une matrice hexagonale ---")
    
    # 5.1 Créer une matrice hexagonale
    print("\n5.1 Création d'une matrice hexagonale de microlentilles")
    microlens_array_hex = create_microlens_array(
        name="Microlentilles hexagonales",
        pitch_mm=0.5,
        num_elements_x=7,
        num_elements_y=7,
        focal_length_mm=20.0,
        array_pattern=ArrayPattern.HEXAGONAL,
        wavelength_nm=wavelength_nm
    )
    
    # 5.2 Créer une caméra adaptée
    camera_hex = RealCamera(
        name="Camera hexagonale",
        num_pixels_x=512,
        num_pixels_y=512,
        pixel_size_um=5.0,
        wavelength_nm=wavelength_nm
    )
    
    # 5.3 Créer le système Shack-Hartmann
    sh_hex = ShackHartmann(
        name="ShackHartmann_Hexagonal",
        microlens_array=microlens_array_hex,
        camera=camera_hex,
        wavelength_nm=wavelength_nm,
        focal_length_mm=20.0,
        centroid_algorithm=CentroidAlgorithm.WEIGHTED_CENTROID,
        display=True,
        display_dir=output_dir
    )
    
    print(f"  {sh_hex}")
    print(f"  Motif de la matrice: {microlens_array_hex.array_pattern.value}")
    
    # 5.4 Simuler avec un faisceau
    print("\n5.4 Simulation Shack-Hartmann hexagonale")
    beam_hex = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        num_points=512
    )
    electric_field_hex = beam_hex.generate_electric_field(
        method="gaussian",
        sigma_mm=1.0
    )
    beam_hex.electric_field = electric_field_hex
    beam_hex.intensity = beam_hex.compute_intensity_from_electric_field(electric_field_hex)
    
    sh_hex.simulate(beam_hex)
    
    # 5.5 Statistiques
    print("\n5.5 Statistiques des résultats")
    stats_hex = sh_hex.get_slope_statistics()
    
    print(f"  Pentes X: PV={stats_hex['slopes_x']['pv']:.6f}, RMS={stats_hex['slopes_x']['rms']:.6f} rad/mm")
    print(f"  Pentes Y: PV={stats_hex['slopes_y']['pv']:.6f}, RMS={stats_hex['slopes_y']['rms']:.6f} rad/mm")
    
    # 5.6 Visualiser les résultats
    print("\n5.6 Visualisation des résultats")
    sh_hex.visualize_all(output_dir=output_dir)
    
    # =========================================================================
    # 6. SIMULATION AVEC UNE CAMÉRA RÉELLE (BRUIT)
    # =========================================================================
    print("\n--- 6. Simulation avec une caméra réelle (bruit) ---")
    
    # 6.1 Créer une caméra avec du bruit
    print("\n6.1 Création d'une caméra réelle avec bruit")
    noisy_camera = RealCamera(
        name="Camera bruyante",
        num_pixels_x=512,
        num_pixels_y=512,
        pixel_size_um=5.0,
        wavelength_nm=wavelength_nm,
        material_name="Silicon",
        quantum_efficiency=0.8,
        full_well_capacity=50000,
        readout_noise_e=10.0,  # Bruit de lecture élevé
        dark_current_e=0.5,    # Courant d'obscurité
        exposure_time_s=0.1
    )
    
    # 6.2 Créer le système Shack-Hartmann avec la caméra bruyante
    sh_noisy = ShackHartmann(
        name="ShackHartmann_Noisy",
        microlens_array=sh_default.microlens_array,
        camera=noisy_camera,
        wavelength_nm=wavelength_nm,
        focal_length_mm=20.0,
        centroid_algorithm=CentroidAlgorithm.WEIGHTED_CENTROID,
        display=True,
        display_dir=output_dir
    )
    
    # 6.3 Simuler avec un faisceau
    print("\n6.3 Simulation avec une caméra bruyante")
    beam_noisy = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        num_points=512
    )
    electric_field_noisy = beam_noisy.generate_electric_field(
        method="gaussian",
        sigma_mm=1.0
    )
    beam_noisy.electric_field = electric_field_noisy
    beam_noisy.intensity = beam_noisy.compute_intensity_from_electric_field(electric_field_noisy)
    
    sh_noisy.simulate(beam_noisy)
    
    # 6.4 Statistiques
    print("\n6.4 Statistiques des résultats (caméra bruyante)")
    stats_noisy = sh_noisy.get_slope_statistics()
    
    print(f"  Pentes X: PV={stats_noisy['slopes_x']['pv']:.6f}, RMS={stats_noisy['slopes_x']['rms']:.6f} rad/mm")
    print(f"  Pentes Y: PV={stats_noisy['slopes_y']['pv']:.6f}, RMS={stats_noisy['slopes_y']['rms']:.6f} rad/mm")
    print(f"  Erreurs sur les pentes X: Moyenne={stats_noisy['slope_errors_x']['mean']:.6f} rad/mm")
    print(f"  Erreurs sur les pentes Y: Moyenne={stats_noisy['slope_errors_y']['mean']:.6f} rad/mm")
    
    # 6.5 Comparaison avec la caméra idéale
    print("\n6.5 Comparaison avec la caméra idéale")
    
    print(f"\n  Caméra idéale:")
    print(f"    Erreurs sur les pentes X: Moyenne={stats_default['slope_errors_x']['mean']:.6f} rad/mm")
    print(f"    Erreurs sur les pentes Y: Moyenne={stats_default['slope_errors_y']['mean']:.6f} rad/mm")
    
    print(f"\n  Caméra bruyante:")
    print(f"    Erreurs sur les pentes X: Moyenne={stats_noisy['slope_errors_x']['mean']:.6f} rad/mm")
    print(f"    Erreurs sur les pentes Y: Moyenne={stats_noisy['slope_errors_y']['mean']:.6f} rad/mm")
    
    print(f"\n  Ratio des erreurs (bruyante/idéale):")
    print(f"    Pentes X: {stats_noisy['slope_errors_x']['mean'] / stats_default['slope_errors_x']['mean']:.2f}")
    print(f"    Pentes Y: {stats_noisy['slope_errors_y']['mean'] / stats_default['slope_errors_y']['mean']:.2f}")
    
    # 6.6 Visualiser les résultats
    print("\n6.6 Visualisation des résultats")
    sh_noisy.visualize_all(output_dir=output_dir)
    
    # =========================================================================
    # 7. EXPORT DES DONNÉES
    # =========================================================================
    print("\n--- 7. Export des données ---")
    
    # 7.1 Exporter les données du faisceau aberré
    print("\n7.1 Export des données du faisceau aberré")
    data_aberrated = sh_aberrated.get_slope_data()
    
    # Sauvegarder en JSON
    output_file = os.path.join(output_dir, "shack_hartmann_aberrated_data.json")
    with open(output_file, 'w') as f:
        json_data = {
            'name': data_aberrated['name'],
            'wavelength_nm': data_aberrated['wavelength_nm'],
            'focal_length_mm': data_aberrated['focal_length_mm'],
            'pixel_size_mm': data_aberrated['pixel_size_mm'],
            'slopes_x': data_aberrated['slopes_x'].tolist(),
            'slopes_y': data_aberrated['slopes_y'].tolist(),
            'slope_errors_x': data_aberrated['slope_errors_x'].tolist(),
            'slope_errors_y': data_aberrated['slope_errors_y'].tolist(),
            'centroids': data_aberrated['centroids'].tolist(),
            'centroid_errors': data_aberrated['centroid_errors'].tolist(),
            'statistics': stats_aberrated
        }
        json.dump(json_data, f, indent=2)
    
    print(f"  ✓ Données sauvegardées dans {output_file}")
    
    # 7.2 Exporter les données de comparaison
    print("\n7.2 Export des données de comparaison")
    comparison_data = {
        'algorithms': {}
    }
    
    for algo, algo_name in algorithms:
        comparison_data['algorithms'][algo.value] = {
            'name': algo_name,
            'data': results[algo.value]
        }
    
    comparison_file = os.path.join(output_dir, "shack_hartmann_algorithms_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"  ✓ Données sauvegardées dans {comparison_file}")
    
    # =========================================================================
    # 8. SIMULATION AVEC DIFFÉRENTES TAILLES DE TÂCHES
    # =========================================================================
    print("\n--- 8. Simulation avec différentes tailles de tâches ---")
    
    # 8.1 Taille de tâche standard (sigma=1.0 mm)
    print("\n8.1 Taille de tâche standard (σ=1.0 mm)")
    sh_std = create_shack_hartmann(
        name="ShackHartmann_StdSpot",
        num_microlenses_x=5,
        num_microlenses_y=5,
        focal_length_mm=20.0,
        display=False
    )
    
    beam_std = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        num_points=256
    )
    electric_field_std = beam_std.generate_electric_field(
        method="gaussian",
        sigma_mm=1.0
    )
    beam_std.electric_field = electric_field_std
    beam_std.intensity = beam_std.compute_intensity_from_electric_field(electric_field_std)
    
    sh_std.simulate(beam_std)
    stats_std = sh_std.get_slope_statistics()
    
    print(f"  Pentes X: PV={stats_std['slopes_x']['pv']:.6f}, RMS={stats_std['slopes_x']['rms']:.6f} rad/mm")
    print(f"  Erreurs sur les centroïdes: Moyenne={np.mean(sh_std.centroid_errors):.4f} pixels")
    
    # 8.2 Taille de tâche petite (sigma=0.5 mm)
    print("\n8.2 Taille de tâche petite (σ=0.5 mm)")
    sh_small = create_shack_hartmann(
        name="ShackHartmann_SmallSpot",
        num_microlenses_x=5,
        num_microlenses_y=5,
        focal_length_mm=20.0,
        display=False
    )
    
    beam_small = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        num_points=256
    )
    electric_field_small = beam_small.generate_electric_field(
        method="gaussian",
        sigma_mm=0.5
    )
    beam_small.electric_field = electric_field_small
    beam_small.intensity = beam_small.compute_intensity_from_electric_field(electric_field_small)
    
    sh_small.simulate(beam_small)
    stats_small = sh_small.get_slope_statistics()
    
    print(f"  Pentes X: PV={stats_small['slopes_x']['pv']:.6f}, RMS={stats_small['slopes_x']['rms']:.6f} rad/mm")
    print(f"  Erreurs sur les centroïdes: Moyenne={np.mean(sh_small.centroid_errors):.4f} pixels")
    
    # 8.3 Taille de tâche grande (sigma=2.0 mm)
    print("\n8.3 Taille de tâche grande (σ=2.0 mm)")
    sh_large = create_shack_hartmann(
        name="ShackHartmann_LargeSpot",
        num_microlenses_x=5,
        num_microlenses_y=5,
        focal_length_mm=20.0,
        display=False
    )
    
    beam_large = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        num_points=256
    )
    electric_field_large = beam_large.generate_electric_field(
        method="gaussian",
        sigma_mm=2.0
    )
    beam_large.electric_field = electric_field_large
    beam_large.intensity = beam_large.compute_intensity_from_electric_field(electric_field_large)
    
    sh_large.simulate(beam_large)
    stats_large = sh_large.get_slope_statistics()
    
    print(f"  Pentes X: PV={stats_large['slopes_x']['pv']:.6f}, RMS={stats_large['slopes_x']['rms']:.6f} rad/mm")
    print(f"  Erreurs sur les centroïdes: Moyenne={np.mean(sh_large.centroid_errors):.4f} pixels")
    
    # 8.4 Tableau comparatif
    print("\n8.4 Tableau comparatif des tailles de tâches")
    print("\n{:<15} {:<15} {:<15} {:<15} {:<15}".format(
        "Taille",
        "PV X (rad/mm)",
        "RMS X (rad/mm)",
        "Erreur centroïde (pixels)"
    ))
    print("-" * 65)
    
    print("{:<15} {:<15.6f} {:<15.6f} {:<15.4f}".format(
        "Petite (σ=0.5)",
        stats_small['slopes_x']['pv'],
        stats_small['slopes_x']['rms'],
        np.mean(sh_small.centroid_errors)
    ))
    
    print("{:<15} {:<15.6f} {:<15.6f} {:<15.4f}".format(
        "Standard (σ=1.0)",
        stats_std['slopes_x']['pv'],
        stats_std['slopes_x']['rms'],
        np.mean(sh_std.centroid_errors)
    ))
    
    print("{:<15} {:<15.6f} {:<15.6f} {:<15.4f}".format(
        "Grande (σ=2.0)",
        stats_large['slopes_x']['pv'],
        stats_large['slopes_x']['rms'],
        np.mean(sh_large.centroid_errors)
    ))
    
    # =========================================================================
    # 9. FIN DE L'EXEMPLE
    # =========================================================================
    print("\n" + "="*80)
    print("Example 13 terminé avec succès !")
    print(f"Les images ont été sauvegardées dans {output_dir}/")
    print("="*80)
    print("\nRésumé des fonctionnalités démontrées:")
    print("  ✓ Création d'un système Shack-Hartmann complet")
    print("  ✓ Simulation avec un faisceau gaussien")
    print("  ✓ Comparaison de 4 algorithmes de calcul des centroïdes")
    print("    - Barycentre pondéré")
    print("    - Barycentre avec seuil")
    print("    - Ajustement gaussien")
    print("    - Basé sur les moments")
    print("  ✓ Calcul des pentes locales de phase (dφ/dx, dφ/dy)")
    print("  ✓ Estimation des erreurs sur les centroïdes et les pentes")
    print("  ✓ Simulation avec un faisceau aberré (Zernike)")
    print("  ✓ Matrice de microlentilles hexagonale")
    print("  ✓ Caméra avec bruit (puits quantiques, bruit de lecture, etc.)")
    print("  ✓ Visualisation :")
    print("    - Carte des tâches d'Airy")
    print("    - Carte des centroïdes")
    print("    - Cartes des pentes locales (X et Y)")
    print("    - Cartes d'erreur (centroïdes et pentes)")
    print("  ✓ Comparaison des algorithmes et des configurations")
    print("  ✓ Export des données en JSON")
    print("="*80)


if __name__ == "__main__":
    os.makedirs('examples/output', exist_ok=True)
    run_shack_hartmann_example()
