# Experimental protocol

## Source and task
Binary pixel segmentation of polyp (1) vs background (0) on the public Kvasir-SEG dataset. 1,000 matched image/mask pairs. No patient-wise split was available in the captured experiment.

## Split / preprocessing
- Random seed: 2026. 800 train / 100 validation / 100 held-out test.
- 256×256 RGB inputs with OpenCV resize (linear interpolation), masks with nearest-neighbor interpolation and `>127` threshold.
- Pixel intensities divided by 255. FPN and DeepLabV3+ additionally use ImageNet mean/std normalization.
- The rare foreground class gets a training-set-derived positive weight in BCE.

## Models
- E0: custom U-Net with weighted BCE.
- E1: E0 plus BatchNorm + 0.10 Dropout2d.
- E2: E1 plus combined 0.5 weighted BCE + 0.5 Dice Loss.
- E3: E2 plus synced horizontal/vertical flips, 90° rotations and brightness changes.
- E4: E3 plus ReduceLROnPlateau (validation IoU).
- E5: E4 plus ConvTranspose2d decoder upsampling.
- Ready-made baselines: FPN and DeepLabV3+ (segmentation_models_pytorch), ImageNet-pretrained ResNet34 encoder.

## Training
- Optimizer: Adam. LR 0.001 (custom U-Net) / 0.0003 (FPN, DeepLabV3+). Batch size: 4.
- Maximum 32 U-Net epochs and 25 external-model epochs. Early stopping with patience 9.
- Best checkpoint selected by validation mean per-image IoU at threshold 0.50; validation loss breaks ties.
- Validation-only threshold grid 0.20, 0.25, ... 0.70. Then the selected checkpoint/threshold is evaluated on the test set.
- `mean IoU`: arithmetic average of per-image intersection-over-union values. `mean Dice`: analogous mean per-image Dice coefficient.

## Results
| Model | Test IoU | Test Dice |
|---|---:|---:|
| E0 U-Net baseline | 0.3718 | 0.4855 |
| E2 U-Net BCE + Dice | 0.6028 | 0.7163 |
| FPN ResNet34 | 0.8302 | 0.8896 |
| DeepLabV3+ ResNet34 | 0.7983 | 0.8682 |

Selected test thresholds were 0.70 for all four in the recorded run; E1 validation threshold was 0.65. These measurements are from the final executed notebook and were not reproduced by GitHub Actions.

## Reproducibility and limitations
The notebook embeds its full training worker for Kaggle import. Checkpoints are written atomically and resumed per epoch in the active working directory. Results can vary with software/GPU versions and stochastic operations. The experiment is not cross-validated, does not guarantee patient-disjoint images, and should not be used for clinical decisions.

Model weights are excluded from this repository to keep source history lightweight; rerun the notebook to create weights and figures. Original dataset is distributed by its authors via Kaggle.