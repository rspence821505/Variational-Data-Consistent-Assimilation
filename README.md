# Variational-Data-Consistent-Assimilation

This repository contains the code to reproduce all figures from the paper Variational Data-Consistent Assimilation on data assimilation methods applied to dynamical systems. The paper compares various cost functions including 4D-Var, Data-Consistent 4D-Var (DC 4D-Var), and Data-Consistent Weighted Mean Error 4D-Var (DCI-WME 4DVar).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Figure Descriptions](#figure-descriptions)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## Overview

This codebase implements and compares three data assimilation methods:

- **4D-Var (Bayes)**: Traditional four-dimensional variational data assimilation
- **DCI**: Data-Consistent 4D-Var
- **DCI-WME**: DCI with Weighted Mean Error QoI Map 4D-Var

The methods are tested on 3 chaotic dynamical systems:

- **Lorenz 63**: A simplified atmospheric convection model
- **Lorenz 96**: A higher-dimensional model simulating atmospheric dynamics
- **Shallow Water Equations**: A two-dimensional nonlinear PDE model often used in coastal modeling that describes the evolution of water height and momentum over time.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/rspence821505/Variational-Data-Consistent-Assimilation.git
   cd Variational-Data-Consistent-Assimilation
   ```

2. **Create a conda environment (recommended):**

   ```bash
   conda create -n data-assimilation python=3.9
   conda activate data-assimilation
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Important Notes

- **JAX Version**: This code requires JAX version 0.5.2 specifically. The installation will fail if a different version is detected.
- **Hardware**: The code is configured to use JAX with CPU. The code does not currently require JAX GPU acceleration.

## Quick Start

Generate all figures from the paper:

```bash
python paper_figures.py
```

This will create a `figures/` directory and generate all 8 figures used in the paper.

## Usage

The `paper_figures.py` script provides flexible options for figure generation:

### Generate All Figures (Default)

```bash
python paper_figures.py
# or explicitly
python paper_figures.py --all
```

### Generate Specific Figures

```bash
# Generate a single figure
python paper_figures.py --figure 3

# Generate multiple specific figures
python paper_figures.py --figures 1 2 5 8

# Generate figures for Lorenz 63 only (figures 1-4)
python paper_figures.py --figures 1 2 3 4
```

### Command Line Options

| Option                | Description                    | Example                                   |
| --------------------- | ------------------------------ | ----------------------------------------- |
| `--all`               | Generate all figures (default) | `python paper_figures.py --all`           |
| `--figure N`          | Generate single figure N (1-8) | `python paper_figures.py --figure 3`      |
| `--figures N1 N2 ...` | Generate multiple figures      | `python paper_figures.py --figures 1 3 5` |

### Output

All figures are saved as PNG files in the `figures/` directory:

## Figure Descriptions

### Lorenz 63 System (Figures 1-4)

| Figure       | Filename                         | Description                                                        |
| ------------ | -------------------------------- | ------------------------------------------------------------------ |
| **Figure 1** | `sigma_b_bound_distribution.png` | Distribution of sigma bounds for background error covariance       |
| **Figure 2** | `obs_inflation_levels_l63.png`   | RMSE heatmap showing performance across noise and inflation levels |
| **Figure 3** | `tarmse_plot_l63.png`            | Time-averaged RMSE comparison across methods                       |
| **Figure 4** | `dci_wme_noisevdensity.png`      | Cross-validation plot showing noise vs. observation density        |

### Lorenz 96 System (Figures 5-8)

| Figure       | Filename                          | Description                                                   |
| ------------ | --------------------------------- | ------------------------------------------------------------- |
| **Figure 5** | `L96_rmse_dof.png`                | RMSE scaling with degrees of freedom (system dimension)       |
| **Figure 6** | `combined_rmse_bias_plot_l96.png` | Combined RMSE and bias analysis for background and analysis   |
| **Figure 7** | `window_size_plot_l96.png`        | Impact of assimilation window size on performance             |
| **Figure 8** | `comp_cost_plot_l96.png`          | Computational cost comparison across methods and system sizes |

## Project Structure

```
.
├── paper_figures.py          # Main script for figure generation
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── configs.py               # Configuration classes for experiments
├── models.py                # Lorenz system implementations
├── metrics.py               # RMSE, bias, and other evaluation metrics
├── plotting.py              # Plotting utilities and functions
├── cost_functions.py        # Implementation of 4D-Var, DC-4DVar, and DC-WME 4DVar
└── figures/                 # Generated figures (created automatically)
    ├── sigma_b_bound_distribution.png
    ├── obs_inflation_levels_l63.png
    ├── tarmse_plot_l63.png
    ├── dci_wme_noisevdensity.png
    ├── L96_rmse_dof.png
    ├── combined_rmse_bias_plot_l96.png
    ├── window_size_plot_l96.png
    └── comp_cost_plot_l96.png
```

## Dependencies

Key dependencies include:

- **JAX 0.5.2**: High-performance numerical computing

See `requirements.txt` for the complete list with version specifications.

## Troubleshooting

### Common Issues

**JAX Version Error:**

```
ValueError: JAX version X.X.X is not supported. Please use JAX version 0.5.2.
```

_Solution:_ Install the specific JAX version:

```bash
pip install jax==0.5.2
```

**Missing Figures Directory:**
The script automatically creates the `figures/` directory, but ensure you have write permissions in the current directory.

**Execution:**

- Figure generation can take several minutes to hours depending on your hardware
- Figures 2, 5, and 8 are particularly computational intensive
- Consider using `--figure N` to generate individual figures during development

### Performance Tips

1. **Generate figures incrementally** during development:

   ```bash
   python paper_figures.py --figure 1  # Quick test
   ```

2. **Monitor progress** with the built-in tqdm progress bars

3. **Use multiple cores** by setting:
   ```bash
   export XLA_FLAGS="--xla_force_host_platform_device_count=8"  # Adjust based on your CPU
   ```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{
  title={Variational Data Consistent Assimilation},
  author={Rylan Spence, Troy Butler, and Clint Dawson},
  journal={TBD},
  year={2025},
  volume={TBD},
  pages={TBD}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the code or paper, please:

- Open an issue on GitHub
- Contact [rylan.spence@utexas.edu](mailto:rylan.spence@utexas.edu)

---

**Note:** This code is provided as-is for reproducibility purposes. While we strive for accuracy, please verify results independently for your own research applications.
