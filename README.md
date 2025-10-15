# PyBOSARIS

**Python port of the BOSARIS Toolkit for speaker recognition and biometric systems evaluation**

PyBOSARIS is a comprehensive Python library for evaluating and calibrating binary classification systems, with a focus on speaker recognition and biometric authentication. It's a faithful Python implementation of the original MATLAB [BOSARIS Toolkit](https://sites.google.com/site/bosaristoolkit/).

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Test Coverage: 68%](https://img.shields.io/badge/coverage-68%25-yellow.svg)]()

## Features

### Core Data Structures ✅
- **Ndx (Index)**: Trial index specification for model-test pairs
- **Key**: Supervised trial labels (target/non-target)
- **Scores**: Score matrices with validation masks

### Calibration Methods ✅
- **PAV (Pool Adjacent Violators)**: Non-parametric isotonic regression calibration (99% coverage)
- **Logistic Regression**: Linear calibration with scaling and offset (100% coverage)
- Separate development/evaluation set support
- Transform scores to well-calibrated log-likelihood ratios

### Evaluation Metrics ✅
- **EER (Equal Error Rate)**: Operating point where miss rate equals false alarm rate
- **minDCF**: Minimum detection cost function
- **actDCF**: Actual DCF at Bayes decision threshold
- **Cllr**: Calibration loss (log-likelihood ratio cost)
- **minCllr**: Minimum Cllr after optimal calibration
- **ROCCH**: ROC Convex Hull using PAV algorithm (92% coverage)

### Score Fusion ✅
- **Linear Fusion**: Weighted combination of multiple systems (97% coverage)
- Optimized using L-BFGS-B algorithm
- Support for same-data and dev/eval splits

### I/O and Visualization ✅
- **HDF5 I/O**: Efficient binary format for save/load (optional: requires h5py)
- **DET Curves**: Detection Error Tradeoff plots (optional: requires matplotlib)
- Probit-scale axes with publication-quality output

## Installation

### Basic Installation
```bash
git clone https://github.com/wa3dbk/PyBOSARIS.git
 cd PyBOSARIS
pip install -e .
```

### With Optional Dependencies
```bash
# For HDF5 I/O support
pip install h5py

# For plotting support
pip install matplotlib

# Install all optional dependencies
pip install h5py matplotlib
```

## Quick Start

### Basic Evaluation

```python
import numpy as np
from pybosaris.core import Key, Scores
from pybosaris.evaluation import compute_eer, compute_min_dcf, compute_cllr

# Create trial structure
tar_mask = np.array([[True, False, False], [False, True, False]])
non_mask = np.array([[False, True, True], [True, False, True]])
key = Key(['model1', 'model2'], ['test1', 'test2', 'test3'],
          tar_mask, non_mask)

# Create scores
score_mat = np.array([[2.5, -1.2, -0.8], [0.3, 3.1, -1.5]])
scores = Scores(['model1', 'model2'], ['test1', 'test2', 'test3'],
                score_mat)

# Extract target and non-target scores
tar_scores, non_scores = scores.get_tar_non(key)

# Compute metrics
eer = compute_eer(tar_scores, non_scores)
min_dcf, _, _ = compute_min_dcf(tar_scores, non_scores)

print(f"EER: {eer*100:.2f}%")
print(f"minDCF: {min_dcf:.4f}")
```

### Score Calibration

```python
from pybosaris.calibration import pav_calibrate_scores, logistic_calibrate_scores

# PAV calibration
pav_calibrated = pav_calibrate_scores(scores, key)

# Logistic regression calibration
lr_calibrated, weights = logistic_calibrate_scores(scores, key)

# Evaluate calibrated scores as LLRs
cal_tar, cal_non = pav_calibrated.get_tar_non(key)
cllr = compute_cllr(cal_tar, cal_non)
print(f"Cllr: {cllr:.4f}")
```

### Score Fusion

```python
from pybosaris.fusion import linear_fuse_scores

# Fuse multiple systems
scores_list = [scores_system1, scores_system2, scores_system3]
fused_scores, fusion_weights = linear_fuse_scores(scores_list, key)

print(f"Fusion weights: {fusion_weights}")
```

### DET Curve Plotting

```python
from pybosaris.plotting import plot_det_curve, plot_det_curves
import matplotlib.pyplot as plt

# Single system
plot_det_curve(tar_scores, non_scores, label='System 1')
plt.show()

# Multiple systems comparison
plot_det_curves(
    [(tar1, non1), (tar2, non2), (tar3, non3)],
    labels=['Acoustic', 'Prosodic', 'Fused'],
    title='System Comparison'
)
plt.savefig('comparison.png', dpi=300)
```

### Save and Load

```python
# Save to HDF5 (requires h5py)
scores.save('scores.h5')
key.save('key.h5')

# Load from HDF5
loaded_scores = Scores.load('scores.h5')
loaded_key = Key.load('key.h5')
```

## Complete Workflow Example

```python
import numpy as np
from pybosaris.core import Key, Scores
from pybosaris.calibration import pav_calibrate_scores
from pybosaris.fusion import linear_fuse_scores
from pybosaris.evaluation import compute_all_metrics
from pybosaris.plotting import plot_det_curves
import matplotlib.pyplot as plt

# 1. Setup trial structure
tar_mask = np.array([
    [True, False, False, True],
    [False, True, False, False],
    [False, False, True, False]
])
non_mask = np.array([
    [False, True, True, False],
    [True, False, True, True],
    [True, True, False, True]
])
key = Key(['m1', 'm2', 'm3'], ['t1', 't2', 't3', 't4'], tar_mask, non_mask)

# 2. Create scores from two systems
scores_acoustic = Scores(['m1', 'm2', 'm3'], ['t1', 't2', 't3', 't4'],
                         np.array([[3.0, -2.0, -1.5, 2.5],
                                  [-0.5, 2.5, -1.0, -1.2],
                                  [-0.8, -1.3, 2.8, -1.5]]))

scores_prosodic = Scores(['m1', 'm2', 'm3'], ['t1', 't2', 't3', 't4'],
                         np.array([[2.5, -1.5, -1.0, 2.0],
                                  [0.0, 2.0, -0.5, -0.8],
                                  [-0.3, -0.8, 2.3, -1.0]]))

# 3. Evaluate raw systems
print("=== Raw Systems ===")
for name, scores in [('Acoustic', scores_acoustic), ('Prosodic', scores_prosodic)]:
    tar, non = scores.get_tar_non(key)
    metrics = compute_all_metrics(tar, non, is_llr=False)
    print(f"{name}: EER={metrics['eer']*100:.2f}%, minDCF={metrics['min_dcf']:.4f}")

# 4. Calibrate both systems
cal_acoustic = pav_calibrate_scores(scores_acoustic, key)
cal_prosodic = pav_calibrate_scores(scores_prosodic, key)

# 5. Fuse calibrated systems
fused_scores, weights = linear_fuse_scores([cal_acoustic, cal_prosodic], key)
print(f"\nFusion weights: {weights}")

# 6. Evaluate fused system
tar_fused, non_fused = fused_scores.get_tar_non(key)
metrics = compute_all_metrics(tar_fused, non_fused, is_llr=True)
print(f"\n=== Fused System ===")
print(f"EER: {metrics['eer']*100:.2f}%")
print(f"minDCF: {metrics['min_dcf']:.4f}")
print(f"Cllr: {metrics['cllr']:.4f}")
print(f"actDCF: {metrics['act_dcf']:.4f}")

# 7. Plot DET curves
tar_ac, non_ac = cal_acoustic.get_tar_non(key)
tar_pr, non_pr = cal_prosodic.get_tar_non(key)

plot_det_curves(
    [(tar_ac, non_ac), (tar_pr, non_pr), (tar_fused, non_fused)],
    labels=['Acoustic (cal)', 'Prosodic (cal)', 'Fused'],
    title='System Performance Comparison'
)
plt.savefig('system_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Project Status

**Current Version: 1.0.0**

### Development Progress

- [x] **Phase 1**: Project Setup & Architecture
- [x] **Phase 2**: Core Data Structures (Ndx, Key, Scores) - **100% complete**
- [x] **Phase 3**: Calibration Algorithms (PAV, Logistic) - **100% complete**
- [x] **Phase 4**: Fusion Methods - **97% complete**
- [x] **Phase 5**: Evaluation Metrics - **92% complete**
- [x] **Phase 7**: Plotting & Visualization - **Implemented**
- [x] **Phase 8**: File I/O (HDF5) - **Implemented**
- [x] **Phase 11**: Testing & Validation - **172 tests, 68% coverage**

### Test Coverage by Module

| Module | Coverage | Tests |
|--------|----------|-------|
| Calibration (PAV) | 99% | 40 |
| Calibration (Logistic) | 100% | 29 |
| Evaluation Metrics | 92% | 38 |
| Fusion (Linear) | 97% | 23 |
| Core (Ndx) | 100% | 18 |
| Math Utils | 97% | 14 |
| **Overall** | **68%** | **172** |

## Architecture

```
pybosaris/
├── core/              # Data structures (Ndx, Key, Scores)
├── calibration/       # PAV and logistic regression
├── evaluation/        # Performance metrics
├── fusion/            # Score fusion algorithms
├── io/                # File I/O (HDF5)
├── plotting/          # Visualization (DET curves)
└── utils/             # Math functions and validation
```

## Dependencies

### Required
- Python ≥ 3.9
- NumPy ≥ 1.20
- SciPy ≥ 1.7

### Optional
- h5py ≥ 3.0 (for HDF5 I/O)
- matplotlib ≥ 3.3 (for plotting)

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pybosaris --cov-report=html

# Run specific test file
pytest tests/test_evaluation.py -v
```

## Comparison with Original BOSARIS

PyBOSARIS provides Python implementations of core BOSARIS functionality:

| Feature | MATLAB BOSARIS | PyBOSARIS | Status |
|---------|----------------|-----------|--------|
| Core data structures | ✓ | ✓ | **Complete** |
| PAV calibration | ✓ | ✓ | **Complete** |
| Logistic calibration | ✓ | ✓ | **Complete** |
| EER, minDCF | ✓ | ✓ | **Complete** |
| Cllr, minCllr | ✓ | ✓ | **Complete** |
| actDCF | ✓ | ✓ | **Complete** |
| Linear fusion | ✓ | ✓ | **Complete** |
| DET plots | ✓ | ✓ | **Complete** |
| Quality-based fusion | ✓ | ⏳ | Planned |
| NIST file formats | ✓ | ⏳ | Planned |
| Tippet plots | ✓ | ⏳ | Planned |

## Performance Metrics Explained

### Equal Error Rate (EER)
The operating point where the miss rate equals the false alarm rate. Lower is better.
- Range: [0, 1] (often reported as percentage)
- EER = 0% means perfect separation
- EER = 50% means random performance

### Minimum Detection Cost Function (minDCF)
The minimum cost achievable with an optimal threshold, weighted by prior and costs.
- Range: [0, 1] when normalized
- Depends on prior probability and cost parameters
- Independent of calibration (only measures discrimination)

### Actual DCF (actDCF)
The cost at the Bayes decision threshold for LLR scores.
- Measures both discrimination and calibration
- For well-calibrated LLRs: actDCF ≈ minDCF
- Large difference indicates poor calibration

### Calibration Loss (Cllr)
Log-likelihood ratio cost measuring both discrimination and calibration.
- Range: [0, ∞)
- Cllr = 0 means perfect calibrated LLRs
- Cllr = 1 means scores provide no information

### Minimum Cllr (minCllr)
Minimum Cllr after optimal PAV calibration (discrimination only).
- Measures discrimination independent of calibration
- minCllr ≤ Cllr always
- Difference (Cllr - minCllr) measures calibration loss

## Citation

If you use PyBOSARIS in your research, please cite:

```bibtex
@software{pybosaris2025,
  title = {PyBOSARIS: Python port of the BOSARIS Toolkit},
  author = {Waad Ben Kheder},
  year = {2025},
  url = {https://github.com/wa3dbk/PyBOSARIS}
}
```

Original BOSARIS Toolkit:
```bibtex
@techreport{brummer2011bosaris,
  title = {The BOSARIS Toolkit User Guide},
  author = {Brümmer, Niko and de Villiers, Edward},
  institution = {AGNITIO Research},
  year = {2011}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

PyBOSARIS is inspired by and aims for compatibility with the original MATLAB BOSARIS Toolkit by Niko Brümmer and Edward de Villiers. Special thanks to the speaker recognition research community for developing these evaluation methodologies.
