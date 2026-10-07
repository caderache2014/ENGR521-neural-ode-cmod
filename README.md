# Hybrid Neural ODEs for Alcator C-Mod Plasma Dynamics

A reproduction and extension of the hybrid neural ODE model of Wang et al. (2023) for tokamak plasma dynamics, using experimental data from the Alcator C-Mod tokamak. The model was implemented independently in three frameworks: **Julia SciML**, **JAX**, and **PyTorch**.

This was a graduate research project for ENGR 521 at the University of Washington (Spring 2026), the project course paired with ENGR 520, Physics-Informed Machine Learning.

**Report:** [link to paper]  ·  **Slides:** [link to slides]

## Background

[One or two sentences on what Wang et al. (2023) did and why it matters: e.g., what the hybrid model predicts, which signals it uses, and what makes it "hybrid" (physics-based equations combined with a learned neural network term).]

Reference: [full citation for Wang et al. (2023)]

## Key finding

Beyond reproducing the original results, we asked whether the learned part of the model could be replaced with an interpretable equation using symbolic regression. A diagnostic of the trained model showed that it fits the loop-voltage signal rather than learning a governing physical law for it, which rules out symbolic regression as a path to an interpretable model. A separate sparse-regression analysis with PySINDy, which retained roughly 15 terms rather than a compact equation, supported the same conclusion.

[Optional: one sentence on the noise-limited loop-voltage channel, if that's how the report frames it.]

## Repository structure

| Folder | Implementation | Author |
|---|---|---|
| `Julia_SciML/` | Julia, using the SciML ecosystem | Chris Billingham |
| `JAX-implementation/` | JAX | Phil [surname] |
| `torch-tokode/` | PyTorch | Tino [surname] |

Shared data-handling scripts at the top level:

- `open_cmod_pickle.py` loads the C-Mod dataset.
- `select_viable_romero_data.py` selects the shots used for training and evaluation.

## Data

**The C-Mod experimental data is not included in this repository.** It was shared with the project team by the authors of the original study and is not ours to redistribute. Researchers interested in the data should contact the original authors.

The scripts expect the shot files in a folder named `romero_shots_489/` at the top level of the repository. That folder is listed in `.gitignore` so it can't be committed by accident.

## Running the code

[Brief instructions for each implementation: e.g., the Julia version and how to instantiate the environment, Python package requirements for the JAX and PyTorch versions, and which notebook or script to run first.]

## Team

- **Chris Billingham**: project concept and organization, data acquisition, Julia SciML implementation, model diagnostic
- **Phil [surname]**: JAX implementation, PySINDy analysis
- **Tino [surname]**: PyTorch implementation

Instructor: [name], University of Washington

## Acknowledgments

We thank Allen Wang (MIT) for sharing the Alcator C-Mod dataset used in this project.

## License

Code is released under the MIT License (see `LICENSE`). The license covers the code only, not the experimental data.
