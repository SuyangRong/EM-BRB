# 📌 EM-BRB: An Expectation-Maximization Belief Rule Base for Fault Diagnosis
*Official MATLAB implementation for ICONIP 2026*

## Environment Requirements
- MATLAB R2018a or later
- Statistics and Machine Learning Toolbox

---

## Quick Start
1. Place all files in the same folder and set it as the MATLAB working directory.
2. Run the main script in the MATLAB command window:

```matlab
main_em_brb

##Core Pipeline
Load dataset (automatic 80/20 split if no test set is provided)
PSO optimization for structural parameters
EM algorithm for parameter learning
VEM-Dirichlet rule pruning
Model evaluation and result saving

##Dataset Replacement Guide
When using a new dataset, modify these parameters in main_em_brb.m:
M: Number of input features (default: 2)
N_class: Number of classes (default: 5)
labels: Class label definition (default: 0:0.25:1)

##Citation
If you use this code, please cite our ICONIP 2026 paper.

##License
This project is released under the MIT License.
