"""
Example 12: Camera.py - Virtual Sensors
FR: Exemple complet d'utilisation du module Camera.py.
    Démonstration de :
    - Création de capteurs parfaits (IdealCamera)
    - Création de capteurs réels (RealCamera) avec :
        * Puits quantiques (full well capacity)
        * Bruits standards (gaussien, Poisson, lecture, obscurité, thermique, quantification)
        * Matériau et expansion thermique
        * Filtre couleur (CFA)
    - Échantillonnage de faisceaux
    - Affichage des images avec colormap "hot" pour l'intensité
    - Calcul et affichage du PV (Peak-to-Valley) et du RMS
    
    Unités :
    - Longueurs : mm (taille du capteur), µm (taille des pixels)
    - Longueur d'onde : nm
    - Phase : nm (principale), λ (longueur d'onde), rad, mrad
    - Intensité : a.u. (arbitrary units) ou ADU

EN: Complete example of using Camera.py module.
    Demonstrates:
    - Creation of ideal sensors (IdealCamera)
    - Creation of real sensors (RealCamera) with:
        * Quantum wells (full well capacity)
        * Standard noise (Gaussian, Poisson, readout, dark current, thermal, quantization)
        * Material and thermal expansion
        * Color Filter Array (CFA)
    - Beam sampling
    - Image display with "hot" colormap for intensity
    - PV (Peak-to-Valley) and RMS calculation and display
    
    Units:
    - Lengths: mm (sensor size), µm (pixel size)
    - Wavelength: nm
    - Phase: nm (main), λ (wavelength), rad, mrad
    - Intensity: a.u. (arbitrary units) or ADU

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - matplotlib
    - Camera.py
    - Beam.py
    - Material_Behaviour.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from Camera import (
    IdealCamera, RealCamera, create_camera, SensorType, NoiseType,
    ColorFilterArray, ResponseType, CameraSpecifications
)
from Beam import Beam
from Material_Behaviour import STANDARD_TEMPERATURE_K


def run_camera_example():
    """
    FR: Exécute l'exemple complet de Camera.py.
        Démonstration de toutes les fonctionnalités du module.
    """
    print("\n" + "="*80)
    print("Example 12: Camera.py - Virtual Sensors")
    print("="*80)
    
    # Paramètres globaux
    wavelength_nm = 633.0
    output_dir = 'examples/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # 1. CAPTEUR PARFAIT / IDEAL SENSOR
    # =========================================================================
    print("\n--- 1. Capteur parfait ---")
    
    # 1.1 Création d'un capteur parfait
    print("\n1.1 Création d'un capteur parfait (1024x1024, 5 µm/pixel)")
    ideal_camera = IdealCamera(
        name="Capteur parfait",
        num_pixels_x=1024,
        num_pixels_y=1024,
        pixel_size_um=5.0,  # 5 µm
        wavelength_nm=wavelength_nm,
        display=True,
        display_dir=output_dir
    )
    
    print(f"  {ideal_camera}")
    print(f"  Taille du capteur: {ideal_camera.sensor_width_mm:.2f} x {ideal_camera.sensor_height_mm:.2f} mm")
    print(f"  Nombre de pixels: {ideal_camera.specifications.num_pixels_x} x {ideal_camera.specifications.num_pixels_y}")
    print(f"  Taille des pixels: {ideal_camera.pixel_width_mm*1e3:.1f} x {ideal_camera.pixel_height_mm*1e3:.1f} µm")
    
    # 1.2 Échantillonnage d'un faisceau gaussien
    print("\n1.2 Échantillonnage d'un faisceau gaussien")
    beam_ideal = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.12,  # 1024 * 5 µm = 5.12 mm
        num_points=1024
    )
    electric_field_ideal = beam_ideal.generate_electric_field(method="gaussian", sigma_mm=1.0)
    beam_ideal.electric_field = electric_field_ideal
    beam_ideal.intensity = beam_ideal.compute_intensity_from_electric_field(electric_field_ideal)
    beam_ideal.phase = beam_ideal.extract_phase_from_electric_field(electric_field_ideal)
    
    # Calculer PV et RMS du faisceau initial
    pv_initial, rms_initial = beam_ideal.compute_pv_rms(beam_ideal.intensity)
    print(f"  Faisceau initial: PV={pv_initial:.2f}, RMS={rms_initial:.2f}")
    
    # Échantillonner le faisceau
    image_ideal = ideal_camera.sample_beam(beam_ideal)
    
    # Afficher l'image avec colormap "hot"
    ideal_camera.display_image(
        image_ideal,
        title=f"Capteur parfait - Faisceau gaussien\nPV={pv_initial:.2f}, RMS={rms_initial:.2f}",
        save_path=f'{output_dir}/example12_ideal_camera_beam.png',
        cmap="hot"
    )
    print(f"  ✓ Image sauvegardée: example12_ideal_camera_beam.png")
    
    # Vérification : l'image doit être normalisée (0-1)
    assert np.all(image_ideal >= 0) and np.all(image_ideal <= 1.0), \
        "L'image d'un capteur parfait doit être normalisée entre 0 et 1"
    print(f"  ✓ Vérification: image normalisée entre 0 et 1")
    
    # =========================================================================
    # 2. CAPTEUR RÉEL / REAL SENSOR
    # =========================================================================
    print("\n--- 2. Capteur réel ---")
    
    # 2.1 Création d'un capteur réel
    print("\n2.1 Création d'un capteur réel (512x512, 10 µm/pixel)")
    real_camera = RealCamera(
        name="Capteur réel",
        num_pixels_x=512,
        num_pixels_y=512,
        pixel_size_um=10.0,  # 10 µm
        material_name="Silicon",
        quantum_efficiency=0.9,
        full_well_capacity=100000,
        readout_noise_e=5.0,
        dark_current_e=0.1,
        exposure_time_s=0.1,
        gain_e_per_adu=1.0,
        bit_depth=16,
        cfa=ColorFilterArray.NONE,
        wavelength_nm=wavelength_nm,
        temperature_K=STANDARD_TEMPERATURE_K,
        display=True,
        display_dir=output_dir
    )
    
    print(f"  {real_camera}")
    print(f"  Efficacité quantique: {real_camera.specifications.quantum_efficiency:.1f}")
    print(f"  Capacité des puits: {real_camera.specifications.full_well_capacity} électrons")
    print(f"  Bruit de lecture: {real_camera.specifications.readout_noise_e:.1f} électrons RMS")
    print(f"  Courant d'obscurité: {real_camera.specifications.dark_current_e:.1f} électrons/pixel/s")
    print(f"  Temps d'exposition: {real_camera.specifications.exposure_time_s:.2f} s")
    print(f"  Profondeur de bits: {real_camera.specifications.bit_depth} bits")
    
    # 2.2 Échantillonnage avec bruit
    print("\n2.2 Échantillonnage d'un faisceau gaussien avec bruit")
    beam_real = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.12,  # 512 * 10 µm = 5.12 mm
        num_points=512
    )
    electric_field_real = beam_real.generate_electric_field(method="gaussian", sigma_mm=1.0)
    beam_real.electric_field = electric_field_real
    beam_real.intensity = beam_real.compute_intensity_from_electric_field(electric_field_real)
    beam_real.phase = beam_real.extract_phase_from_electric_field(electric_field_real)
    
    image_real = real_camera.sample_beam(beam_real)
    
    # Calculer PV et RMS de l'image réelle
    pv_real = np.max(image_real) - np.min(image_real)
    rms_real = np.std(image_real)
    mean_real = np.mean(image_real)
    
    print(f"  Image réelle: PV={pv_real:.2f} ADU, RMS={rms_real:.2f} ADU, Mean={mean_real:.2f} ADU")
    
    # Afficher l'image
    real_camera.display_image(
        image_real,
        title=f"Capteur réel - Faisceau gaussien avec bruit\nPV={pv_real:.2f}, RMS={rms_real:.2f}",
        save_path=f'{output_dir}/example12_real_camera_beam.png',
        cmap="hot"
    )
    print(f"  ✓ Image sauvegardée: example12_real_camera_beam.png")
    
    # Vérification : l'image réelle peut dépasser 1.0 à cause du bruit
    assert np.all(image_real >= 0), "L'image doit être positive"
    print(f"  ✓ Vérification: image positive")
    
    # 2.3 Ajout de bruit supplémentaire
    print("\n2.3 Ajout de bruit gaussien supplémentaire (σ=10 ADU)")
    noisy_image = real_camera.add_noise(image_real, NoiseType.GAUSSIAN, sigma=10.0)
    
    pv_noisy = np.max(noisy_image) - np.min(noisy_image)
    rms_noisy = np.std(noisy_image)
    
    print(f"  Image avec bruit supplémentaire: PV={pv_noisy:.2f} ADU, RMS={rms_noisy:.2f} ADU")
    
    real_camera.display_image(
        noisy_image,
        title=f"Capteur réel - Avec bruit gaussien supplémentaire\nPV={pv_noisy:.2f}, RMS={rms_noisy:.2f}",
        save_path=f'{output_dir}/example12_real_camera_noisy.png',
        cmap="hot"
    )
    print(f"  ✓ Image sauvegardée: example12_real_camera_noisy.png")
    
    # =============================================================================
    # 3. DILATATION THERMIQUE / THERMAL EXPANSION
    # =============================================================================
    print("\n--- 3. Dilatation thermique ---")
    
    # 3.1 Calcul des informations de dilatation
    print("\n3.1 Calcul des informations de dilatation thermique (ΔT = +50 K)")
    thermal_info = real_camera.get_thermal_expansion_info(STANDARD_TEMPERATURE_K + 50)
    
    print(f"  Informations de dilatation pour ΔT = +50 K:")
    print(f"    Δpixel_size: {thermal_info['delta_pixel_size_um']:.4f} µm")
    print(f"    Nouveau pixel_size: {thermal_info['new_pixel_size_um']:.4f} µm")
    print(f"    Δsensor_width: {thermal_info['delta_sensor_width_mm']:.6f} mm")
    print(f"    Nouveau sensor_width: {thermal_info['new_sensor_width_mm']:.6f} mm")
    
    # 3.2 Application de la dilatation thermique
    print("\n3.2 Application de la dilatation thermique (ΔT = +50 K)")
    initial_pixel_size_um = real_camera.specifications.pixel_size_um
    initial_sensor_width_mm = real_camera.sensor_width_mm
    
    real_camera.apply_thermal_expansion(STANDARD_TEMPERATURE_K + 50)
    
    new_pixel_size_um = real_camera.specifications.pixel_size_um
    new_sensor_width_mm = real_camera.sensor_width_mm
    
    print(f"  Avant: pixel_size={initial_pixel_size_um:.2f} µm, sensor_width={initial_sensor_width_mm:.2f} mm")
    print(f"  Après:  pixel_size={new_pixel_size_um:.2f} µm, sensor_width={new_sensor_width_mm:.2f} mm")
    
    # Vérifications
    assert new_pixel_size_um > initial_pixel_size_um, "La taille des pixels doit augmenter"
    assert new_sensor_width_mm > initial_sensor_width_mm, "La taille du capteur doit augmenter"
    print(f"  ✓ Toutes les vérifications passées")
    
    # 3.3 Échantillonnage après dilatation
    print("\n3.3 Échantillonnage après dilatation thermique")
    
    # Créer un nouveau faisceau adapté à la nouvelle taille du capteur
    beam_after_thermal = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=real_camera.sensor_width_mm,
        num_points=512
    )
    electric_field_after = beam_after_thermal.generate_electric_field(method="gaussian", sigma_mm=1.0)
    beam_after_thermal.electric_field = electric_field_after
    beam_after_thermal.intensity = beam_after_thermal.compute_intensity_from_electric_field(electric_field_after)
    beam_after_thermal.phase = beam_after_thermal.extract_phase_from_electric_field(electric_field_after)
    
    image_after_thermal = real_camera.sample_beam(beam_after_thermal)
    
    pv_after_thermal = np.max(image_after_thermal) - np.min(image_after_thermal)
    rms_after_thermal = np.std(image_after_thermal)
    
    print(f"  Image après dilatation: PV={pv_after_thermal:.2f} ADU, RMS={rms_after_thermal:.2f} ADU")
    
    real_camera.display_image(
        image_after_thermal,
        title=f"Capteur réel après dilatation - Faisceau gaussien\nPV={pv_after_thermal:.2f}, RMS={rms_after_thermal:.2f}",
        save_path=f'{output_dir}/example12_real_camera_after_thermal.png',
        cmap="hot"
    )
    print(f"  ✓ Image sauvegardée: example12_real_camera_after_thermal.png")
    
    # =========================================================================
    # 4. FILTRE COULEUR (CFA) / COLOR FILTER ARRAY
    # =========================================================================
    print("\n--- 4. Filtre couleur (CFA) ---")
    
    # 4.1 Création d'un capteur avec filtre Bayer RGGB
    print("\n4.1 Capteur avec filtre Bayer RGGB")
    cfa_camera = RealCamera(
        name="Capteur CFA",
        num_pixels_x=64,
        num_pixels_y=64,
        pixel_size_um=10.0,
        cfa=ColorFilterArray.BAYER_RGGB,
        wavelength_nm=wavelength_nm,
        display=True,
        display_dir=output_dir
    )
    
    print(f"  Filtre couleur: {cfa_camera.specifications.cfa.value}")
    
    # 4.2 Échantillonnage avec CFA
    print("\n4.2 Échantillonnage avec filtre Bayer")
    beam_cfa = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=0.64,  # 64 * 10 µm = 0.64 mm
        num_points=64
    )
    electric_field_cfa = beam_cfa.generate_electric_field(method="uniform")
    beam_cfa.electric_field = electric_field_cfa
    beam_cfa.intensity = beam_cfa.compute_intensity_from_electric_field(electric_field_cfa)
    beam_cfa.phase = beam_cfa.extract_phase_from_electric_field(electric_field_cfa)
    
    image_cfa = cfa_camera.sample_beam(beam_cfa)
    
    pv_cfa = np.max(image_cfa) - np.min(image_cfa)
    rms_cfa = np.std(image_cfa)
    
    print(f"  Image avec CFA: PV={pv_cfa:.2f} ADU, RMS={rms_cfa:.2f} ADU")
    
    cfa_camera.display_image(
        image_cfa,
        title=f"Capteur avec CFA Bayer RGGB\nPV={pv_cfa:.2f}, RMS={rms_cfa:.2f}",
        save_path=f'{output_dir}/example12_cfa_camera.png',
        cmap="hot"
    )
    print(f"  ✓ Image sauvegardée: example12_cfa_camera.png")
    
    # =========================================================================
    # 5. RÉPONSE DU CAPTEUR / SENSOR RESPONSE
    # =========================================================================
    print("\n--- 5. Réponse du capteur ---")
    
    # 5.1 Réponse linéaire (par défaut)
    print("\n5.1 Réponse linéaire")
    linear_camera = RealCamera(
        name="Réponse linéaire",
        num_pixels_x=64,
        num_pixels_y=64,
        pixel_size_um=10.0,
        response_type=ResponseType.LINEAR,
        wavelength_nm=wavelength_nm,
        display=False
    )
    
    beam_linear = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=0.64,
        num_points=64
    )
    electric_field_linear = beam_linear.generate_electric_field(method="gaussian", sigma_mm=0.2)
    beam_linear.electric_field = electric_field_linear
    beam_linear.intensity = beam_linear.compute_intensity_from_electric_field(electric_field_linear)
    
    image_linear = linear_camera.sample_beam(beam_linear)
    print(f"  Image linéaire: min={np.min(image_linear):.2f}, max={np.max(image_linear):.2f} ADU")
    
    # 5.2 Réponse logarithmique
    print("\n5.2 Réponse logarithmique")
    log_camera = RealCamera(
        name="Réponse logarithmique",
        num_pixels_x=64,
        num_pixels_y=64,
        pixel_size_um=10.0,
        response_type=ResponseType.LOGARITHMIC,
        wavelength_nm=wavelength_nm,
        display=False
    )
    
    image_log = log_camera.sample_beam(beam_linear)
    print(f"  Image logarithmique: min={np.min(image_log):.2f}, max={np.max(image_log):.2f} ADU")
    
    # 5.3 Réponse avec correction gamma (γ=2.2)
    print("\n5.3 Réponse avec correction gamma (γ=2.2)")
    gamma_camera = RealCamera(
        name="Réponse gamma",
        num_pixels_x=64,
        num_pixels_y=64,
        pixel_size_um=10.0,
        response_type=ResponseType.GAMMA,
        gamma=2.2,
        wavelength_nm=wavelength_nm,
        display=False
    )
    
    image_gamma = gamma_camera.sample_beam(beam_linear)
    print(f"  Image gamma: min={np.min(image_gamma):.2f}, max={np.max(image_gamma):.2f} ADU")
    
    # Visualisation des trois réponses
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    im1 = axes[0].imshow(image_linear, cmap='hot', vmin=0, vmax=np.max(image_linear))
    axes[0].set_title("Réponse linéaire")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(image_log, cmap='hot', vmin=0, vmax=np.max(image_log))
    axes[1].set_title("Réponse logarithmique")
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(image_gamma, cmap='hot', vmin=0, vmax=np.max(image_gamma))
    axes[2].set_title("Réponse gamma (γ=2.2)")
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example12_sensor_responses.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Image sauvegardée: example12_sensor_responses.png")
    
    # =========================================================================
    # 6. TYPES DE BRUIT / NOISE TYPES
    # =========================================================================
    print("\n--- 6. Types de bruit ---")
    
    # Créer une image de base
    base_camera = RealCamera(
        name="Base",
        num_pixels_x=64,
        num_pixels_y=64,
        pixel_size_um=10.0,
        readout_noise_e=0.0,  # Désactiver le bruit de lecture
        dark_current_e=0.0,   # Désactiver le courant d'obscurité
        exposure_time_s=0.0,  # Désactiver le bruit thermique
        wavelength_nm=wavelength_nm,
        display=False
    )
    
    beam_base = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=0.64,
        num_points=64
    )
    electric_field_base = beam_base.generate_electric_field(method="uniform")
    beam_base.electric_field = electric_field_base
    beam_base.intensity = beam_base.compute_intensity_from_electric_field(electric_field_base)
    
    base_image = base_camera.sample_beam(beam_base)
    
    # 6.1 Bruit gaussien
    print("\n6.1 Bruit gaussien (σ=5 ADU)")
    gaussian_noise = base_camera.add_noise(base_image, NoiseType.GAUSSIAN, sigma=5.0)
    
    pv_gaussian = np.max(gaussian_noise) - np.min(gaussian_noise)
    rms_gaussian = np.std(gaussian_noise)
    print(f"  Bruit gaussien: PV={pv_gaussian:.2f} ADU, RMS={rms_gaussian:.2f} ADU")
    
    # 6.2 Bruit de Poisson
    print("\n6.2 Bruit de Poisson")
    poisson_noise = base_camera.add_noise(base_image, NoiseType.POISSON)
    
    pv_poisson = np.max(poisson_noise) - np.min(poisson_noise)
    rms_poisson = np.std(poisson_noise)
    print(f"  Bruit de Poisson: PV={pv_poisson:.2f} ADU, RMS={rms_poisson:.2f} ADU")
    
    # 6.3 Bruit de lecture
    print("\n6.3 Bruit de lecture (5 électrons)")
    readout_noise = base_camera.add_noise(base_image, NoiseType.READOUT, readout_noise_e=5.0)
    
    pv_readout = np.max(readout_noise) - np.min(readout_noise)
    rms_readout = np.std(readout_noise)
    print(f"  Bruit de lecture: PV={pv_readout:.2f} ADU, RMS={rms_readout:.2f} ADU")
    
    # 6.4 Bruit de quantification
    print("\n6.4 Bruit de quantification (8 bits)")
    quantization_noise = base_camera.add_noise(base_image, NoiseType.QUANTIZATION, bit_depth=8)
    
    pv_quant = np.max(quantization_noise) - np.min(quantization_noise)
    rms_quant = np.std(quantization_noise)
    print(f"  Bruit de quantification: PV={pv_quant:.2f} ADU, RMS={rms_quant:.2f} ADU")
    
    # Visualisation des bruits
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    im1 = axes[0, 0].imshow(base_image, cmap='hot')
    axes[0, 0].set_title("Image de base (sans bruit)")
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(gaussian_noise, cmap='hot')
    axes[0, 1].set_title(f"Bruit gaussien\nPV={pv_gaussian:.2f}, RMS={rms_gaussian:.2f}")
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[1, 0].imshow(poisson_noise, cmap='hot')
    axes[1, 0].set_title(f"Bruit de Poisson\nPV={pv_poisson:.2f}, RMS={rms_poisson:.2f}")
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(quantization_noise, cmap='hot')
    axes[1, 1].set_title(f"Bruit de quantification\nPV={pv_quant:.2f}, RMS={rms_quant:.2f}")
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example12_noise_types.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Image sauvegardée: example12_noise_types.png")
    
    # =========================================================================
    # 7. FABRIQUE DE CAPTEURS / CAMERA FACTORY
    # =========================================================================
    print("\n--- 7. Fabrique de capteurs ---")
    
    # 7.1 Création avec la fabrique
    print("\n7.1 Création de capteurs avec create_camera()")
    
    # Capteur parfait
    ideal_from_factory = create_camera(
        camera_type=SensorType.IDEAL,
        name="Capteur parfait (fabrique)",
        num_pixels_x=256,
        num_pixels_y=256,
        pixel_size_um=5.0
    )
    print(f"  Capteur parfait: {ideal_from_factory.specifications.sensor_type.value}")
    
    # Capteur réel
    real_from_factory = create_camera(
        camera_type=SensorType.REAL,
        name="Capteur réel (fabrique)",
        num_pixels_x=256,
        num_pixels_y=256,
        pixel_size_um=10.0,
        material_name="Silicon",
        quantum_efficiency=0.8
    )
    print(f"  Capteur réel: {real_from_factory.specifications.sensor_type.value}")
    
    # =========================================================================
    # 8. INFORMATIONS DU CAPTEUR / SENSOR INFORMATION
    # =========================================================================
    print("\n--- 8. Informations du capteur ---")
    
    # 8.1 Récupération des informations
    print("\n8.1 Récupération des informations du capteur")
    sensor_info = real_camera.get_sensor_info()
    
    print(f"  Informations du capteur '{real_camera.name}':")
    for key, value in sensor_info.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.2f}")
        else:
            print(f"    {key}: {value}")
    
    # 8.2 Échelle des pixels
    print("\n8.2 Échelle des pixels")
    pixel_scale_x, pixel_scale_y = real_camera.get_pixel_scale()
    print(f"  Échelle des pixels: {pixel_scale_x*1e3:.4f} x {pixel_scale_y*1e3:.4f} µm/pixel")
    
    # =========================================================================
    # 9. COMPARAISON CAPTEUR PARFAIT VS RÉEL
    # =========================================================================
    print("\n--- 9. Comparaison capteur parfait vs réel ---")
    
    # Créer un faisceau pour la comparaison
    beam_comparison = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=2.56,  # 512 * 5 µm = 2.56 mm
        num_points=512
    )
    electric_field_comparison = beam_comparison.generate_electric_field(
        method="gaussian", sigma_mm=0.5
    )
    beam_comparison.electric_field = electric_field_comparison
    beam_comparison.intensity = beam_comparison.compute_intensity_from_electric_field(
        electric_field_comparison
    )
    beam_comparison.phase = beam_comparison.extract_phase_from_electric_field(
        electric_field_comparison
    )
    
    # Capteur parfait
    ideal_comparison = IdealCamera(
        name="Parfait",
        num_pixels_x=512,
        num_pixels_y=512,
        pixel_size_um=5.0,
        wavelength_nm=wavelength_nm
    )
    image_ideal_comparison = ideal_comparison.sample_beam(beam_comparison)
    
    # Capteur réel
    real_comparison = RealCamera(
        name="Réel",
        num_pixels_x=512,
        num_pixels_y=512,
        pixel_size_um=5.0,
        material_name="Silicon",
        wavelength_nm=wavelength_nm
    )
    image_real_comparison = real_comparison.sample_beam(beam_comparison)
    
    # Calculer les statistiques
    pv_ideal = np.max(image_ideal_comparison) - np.min(image_ideal_comparison)
    rms_ideal = np.std(image_ideal_comparison)
    
    pv_real = np.max(image_real_comparison) - np.min(image_real_comparison)
    rms_real = np.std(image_real_comparison)
    
    print(f"  Capteur parfait: PV={pv_ideal:.2f}, RMS={rms_ideal:.2f}")
    print(f"  Capteur réel:   PV={pv_real:.2f}, RMS={rms_real:.2f}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(image_ideal_comparison, cmap='hot')
    axes[0].set_title(f"Capteur parfait\nPV={pv_ideal:.2f}, RMS={rms_ideal:.2f}")
    plt.colorbar(im1, ax=axes[0], label="Intensité (a.u.)")
    
    im2 = axes[1].imshow(image_real_comparison, cmap='hot')
    axes[1].set_title(f"Capteur réel\nPV={pv_real:.2f}, RMS={rms_real:.2f}")
    plt.colorbar(im2, ax=axes[1], label="Intensité (ADU)")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example12_ideal_vs_real.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Image sauvegardée: example12_ideal_vs_real.png")
    
    # =========================================================================
    # 10. FIN DE L'EXEMPLE
    # =========================================================================
    print("\n" + "="*80)
    print("Example 12 terminé avec succès !")
    print(f"Les images ont été sauvegardées dans {output_dir}/")
    print("="*80)
    print("\nRésumé des fonctionnalités démontrées:")
    print("  ✓ Création de capteurs parfaits (IdealCamera)")
    print("  ✓ Création de capteurs réels (RealCamera)")
    print("  ✓ Échantillonnage de faisceaux")
    print("  ✓ Ajout de bruit (gaussien, Poisson, lecture, quantification)")
    print("  ✓ Dilatation thermique avec effet sur la taille des pixels")
    print("  ✓ Filtre couleur (CFA)")
    print("  ✓ Réponses du capteur (linéaire, logarithmique, gamma)")
    print("  ✓ Affichage avec colormap 'hot' pour l'intensité")
    print("  ✓ Calcul et affichage du PV et du RMS")
    print("  ✓ Fabrique de capteurs (create_camera)")
    print("="*80)


if __name__ == "__main__":
    os.makedirs('examples/output', exist_ok=True)
    run_camera_example()
