# Hybrid Neural ODEs for Alcator C-Mod Plasma Dynamics

A reproduction and extension of the hybrid neural ODE model of Wang et al. (2023) for tokamak plasma dynamics, using experimental data from the Alcator C-Mod tokamak. The model was implemented independently in three frameworks: **Julia SciML**, **JAX**, and **PyTorch**.

This was a graduate research project for ENGR 521 at the University of Washington (Spring 2026), the project course paired with ENGR 520, Physics-Informed Machine Learning.

**Report:** [PDF](docs/report.pdf)  ·  **Slides:** [PDF](docs/slides.pdf)

## Background

Wang, Garnier, and Rea (2023) applied neural ODEs to predicting the coupled dynamics of plasma current (I<sub>p</sub>) and internal inductance (l<sub>i</sub>) in a tokamak. These quantities matter for safely ramping down a plasma: reducing current tends to raise internal inductance, which is associated with reduced stability. Their starting point was a three-equation ODE model from Romero et al. (2010), in which two equations are exact and the third, governing a summary variable V that would otherwise require evolving PDEs, rests on a physics-motivated approximation. Replacing only that third equation with a neural network while keeping the exact physics produced a hybrid model (RomeroNNV) that outperformed both the original Romero model and a pure neural ODE (MlpODE) on 489 Alcator C-Mod shots.

A. M. Wang, D. T. Garnier, and C. Rea, "Hybridizing Physics and Neural ODEs for Predicting Plasma Inductance Dynamics in Tokamak Fusion Reactors," arXiv:2310.20079 (2023).

Reference: https://arxiv.org/abs/2310.20079 

## Key finding

Beyond reproducing the original results, we asked whether the neural network that replaces the Romero model's empirical dV/dt equation could itself be replaced by a compact, interpretable equation found through symbolic regression. A diagnostic of the trained model showed that the network fit the dV/dt term to the data rather than learning a governing physical law for it, which rules out symbolic regression as a path to an interpretable model. A separate sparse-regression analysis with PySINDy, which retained roughly 15 terms rather than a compact equation, supported the same conclusion.

## Repository structure

| Folder | Implementation | Author |
|---|---|---|
| `Julia_SciML/` | Julia, using the SciML ecosystem | Chris Billingham |
| `JAX-implementation/` | JAX | Phil Prior |
| `torch-tokode/` | PyTorch | Tino Wells |

Shared data-handling scripts at the top level:

- `open_cmod_pickle.py` loads the C-Mod dataset.
- `select_viable_romero_data.py` selects the shots used for training and evaluation.

## Data

**The C-Mod experimental data is not included in this repository.** It was shared with the project team by the authors of the original study and is not ours to redistribute. Researchers interested in the data should contact the original authors.

The scripts expect the shot files in a folder named `romero_shots_489/` at the top level of the repository. That folder is listed in `.gitignore` so it can't be committed by accident.

## Running the code

Each folder contains its own notebooks or scripts; the code expects the data folder described above.

## Team

- **Chris Billingham**: project proposal and organization, data acquisition, Julia SciML implementation, model diagnostic
- **Phil Prior**: JAX implementation, PySINDy analysis
- **Tino Wells**: PyTorch implementation

Instructor: Michelle Hickner, University of Washington - Dept of Mechanical Engineering

## Acknowledgments

We thank Allen Wang (MIT) for sharing the Alcator C-Mod dataset used in this project. We would also 
like to extend our gratitude to Raj Dandekar of Vizuara for first suggesting a re-implementation of Wang et al. (2023) as a starting point to explore symbolic regression and equation discovery.


## License

Code is released under the MIT License (see `LICENSE`). The license covers the code only, not the experimental data.
