EM-BRB: An Expectation-Maximization Belief Rule Base for Fault Diagnosis（ for ICONIP 2026）

---
File Directory
.
├── main_em_brb.m                       # Main entry point (integrates the entire pipeline)
├── em_brb_core.m                       # Core EM algorithm for parameter learning
├── em_brb_predict.m                    # Inference module (disjunctive ER algorithm)
├── compute_activations_em.m            # Calculate rule activation weights
├── fun_gwo_em.m                        # Fitness function for PSO/GWO (NLL)
├── RunPSO.m                            # Particle Swarm Optimizer (default)
├── RunGWO.m                            # Grey Wolf Optimizer (alternative)
├── rule_DirichletVEM_pruning.m         # VEM-Dirichlet sparse pruning (default)
├── rule_lrt_pruning.m                  # LRT likelihood-ratio pruning (alternative)
├── generate_adaptive_reference_points.m# Helper function for adaptive reference points
├── evaluate_metrics_5class.m           # 5-class classification & regression metrics
├── train.txt                           # Training set
├── test.txt                            # Test set
└── em_brb_result.mat                   # Auto-saved results
Environment Requirements
- MATLAB R2018a or later (requires `discretize` function)
- Statistics and Machine Learning Toolbox
Quick Start
1. Place all files in the same folder and set it as MATLAB's Current Folder.
2. Run the main function in MATLAB command window:main_em_brb
Core Pipeline
1. Load dataset (split 80/20 if no test set is provided).
2. PSO optimization for structural parameters (population=25, iterations=120).
3. EM fitting (100 iterations, 3 restarts) for posterior parameter learning.
4. VEM-Dirichlet sparse pruning (α=0.20).
5. Evaluate metrics and save results to `em_brb_result.mat`.
Important Note for Dataset Replacement
Modify these parameters in `main_em_brb.m` when replacing the dataset:
- M: Number of input features (default: 2).
- N_class: Number of classification categories (default: 5).
- labels: Class labels (default: 0:0.25:1).
Citation
If this code contributes to your research, please cite our ICONIP 2026 paper.
License
This project is released under the MIT License.
