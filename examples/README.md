# Examples for SHFromScratch Package

## 📚 Description

This directory contains integration test examples for the **SHFromScratch** package. Each example demonstrates specific features of the package and can be run independently to verify the correct functioning of the optical beam generation, propagation, and visualization modules.

## 📁 Structure

```
examples/
├── __init__.py              # Package initialization
├── README.md                # This file
├── run_all_examples.py      # Script to run all examples
├── output/                  # Directory for output images (created automatically)
├── example_1_gaussian_beam_propagation.py  # Gaussian beam generation and propagation
├── example_2_zernike_beam_propagation.py    # Zernike phase beam generation and propagation
├── example_3_coherence_comparison.py         # Coherent vs incoherent propagation comparison
├── example_4_hermite_gauss_propagation.py   # Hermite-Gauss analytical propagation
├── example_5_laguerre_gauss_propagation.py  # Laguerre-Gauss analytical propagation with OAM
├── example_6_resampling_test.py             # Electric field resampling tests
└── example_7_normalization_test.py          # Phase normalization and units tests
```

## 🚀 How to Run

### Run a Single Example

```bash
python examples/example_1_gaussian_beam_propagation.py
```

### Run All Examples

```bash
python examples/run_all_examples.py
```

**Note:** All examples use `matplotlib` in 'Agg' mode (non-interactive) to avoid display issues. Output images are saved in the `examples/output/` directory.

## 📋 Example Descriptions

### Example 1: Gaussian Beam Propagation
- **Purpose**: Demonstrates basic beam generation and propagation
- **Features tested**:
  - Gaussian beam generation
  - Multiple propagation methods (angular spectrum, Fraunhofer, Fresnel)
  - Different grid sizes (64, 128, 256, 512)
  - Phase and intensity normalization
- **Output**: Initial and propagated beam intensity/phase maps

### Example 2: Zernike Beam Propagation
- **Purpose**: Demonstrates phase generation using Zernike polynomials
- **Features tested**:
  - Zernike phase generation with Noll and Wyant ordering
  - Different numbers of modes (5, 10, 20)
  - Phase normalization (PV vs RMS)
  - Propagation with Zernike phase
- **Output**: Phase maps for different Zernike configurations

### Example 3: Coherence Comparison
- **Purpose**: Compares coherent and incoherent propagation regimes
- **Features tested**:
  - Coherent propagation (complex field)
  - Incoherent propagation (intensity only)
  - Energy conservation
  - Top-hat beam propagation
- **Output**: Intensity maps for coherent vs incoherent propagation

### Example 4: Hermite-Gauss Beam Propagation
- **Purpose**: Demonstrates analytical propagation using Hermite-Gauss modes
- **Features tested**:
  - Modal decomposition on Hermite-Gauss basis
  - Analytical propagation with Gouy phase
  - Comparison with numerical methods
  - Different beam types (Gaussian, super-Gaussian, top-hat)
  - Individual mode visualization
- **Output**: Propagated fields and comparison with numerical methods

### Example 5: Laguerre-Gauss Beam Propagation
- **Purpose**: Demonstrates analytical propagation using Laguerre-Gauss modes
- **Features tested**:
  - Modal decomposition on Laguerre-Gauss basis
  - Orbital Angular Momentum (OAM) beams (l ≠ 0)
  - Analytical propagation
  - Comparison with numerical methods
  - Individual mode visualization
- **Output**: Propagated fields for OAM beams and mode visualizations

### Example 6: Resampling Test
- **Purpose**: Tests electric field resampling between different grid sizes
- **Features tested**:
  - Direct resampling (small to large grid)
  - Propagation with resampling
  - Different grid sizes (32, 64, 128, 256, 512)
  - Energy conservation across grid sizes
  - Diameter change during propagation
- **Output**: Intensity maps before and after resampling

### Example 7: Normalization Test
- **Purpose**: Complete test of normalizations and units
- **Features tested**:
  - Phase normalization (PV vs RMS)
  - Energy units (J, mJ, a.u.)
  - Power units (W, mW, a.u.)
  - Intensity units (W/m², W/cm², a.u.)
  - Energy-power-intensity conversions
  - Impact of normalization on propagation
- **Output**: Phase maps with different normalizations

## 🔧 Requirements

- Python 3.8+
- Required packages (install with `pip install -r requirements.txt`):
  - numpy
  - scipy
  - matplotlib

## 📊 Output

All examples generate output images in the `examples/output/` directory:
- PNG images of intensity and phase maps
- Comparison plots between different methods
- Error metrics and visualizations

## 🎯 Use Cases

These examples serve as:
1. **Integration tests**: Verify that all modules work together correctly
2. **Documentation**: Demonstrate how to use the package features
3. **Validation**: Check that physical quantities (energy, phase) are conserved
4. **Benchmarking**: Compare different propagation methods
5. **Education**: Understand optical beam propagation concepts

## 💡 Tips

- For development: Run examples after making changes to verify functionality
- For debugging: Add `import logging; logging.basicConfig(level=logging.DEBUG)` at the top of any example
- For performance testing: Use larger grid sizes (512, 1024) in example 1 or 6
- For accuracy testing: Compare numerical vs analytical methods in examples 4 and 5

## 📝 Notes

- All examples use the same wavelength (633 nm, red light) by default
- Beam diameter is typically 10 mm
- Propagation distance is typically 500 mm (0.5 m)
- Grid sizes range from 32x32 to 512x512 depending on the test
- Energy is normalized to 1.0 a.u. by default

## 🔄 Version Compatibility

These examples are compatible with:
- `Beam.py` v1.0+
- `Propagation.py` v1.0+
- `Visualization.py` v1.0+
- `MathAndPhysicsTools.py` v1.0+

Last updated: June 19, 2026
