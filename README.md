# Synapse-Sleep-Analysis

Analysis and simulation code for:

> Keita Yamamoto†, Fukuaki L. Kinoshita†, Tomoaki Okami, Daisuke Tone, Hirokazu Sakamoto, Hiroshi Fujishima, Koji L. Ode, Maria Neus Ballester Roig, Ilaria Barone, Nicole M. Gilette, Maria Ericsson, Hiroshi Kiyonari, Eiichi Morii, Kenzo Hirose, Jonathan O. Lipton, Rikuhiro G. Yamada, Hiroki R. Ueda\*.
> **Synaptic Exhaustion Drives Cortical Slow Waves that Encode Wakefulness Levels.**
> *Journal / DOI: to be added.*
> († equal contribution; \* corresponding author)

The study shows that prolonged wakefulness depletes presynaptic vesicles in the prefrontal cortex, that presynaptic exocytotic efficacy bidirectionally regulates NREM slow-wave activity (SWA), and proposes the **EXhausted Information Transfer (EXIT)** model: reduced synaptic efficacy makes postsynaptic firing depend on temporally coincident presynaptic inputs, which drives cortical networks into synchronized slow oscillations. It also introduces **NREM similarity**, an EEG-based metric of wakefulness level.

This repository contains three components:

| Folder | Content | Main figures |
|---|---|---|
| [`01_EM_analysis/`](01_EM_analysis) | Deep-learning segmentation (Detectron2 Cascade Mask R-CNN) and morphometry of presynaptic terminals, active zones (AZs) and synaptic vesicles in transmission electron microscopy (TEM) images | Figs. 1g–j and 3f–h, Extended Data Fig. 1, Extended Data Tables 1–2, Supplementary Tables 1–2 |
| [`02_Simulation/`](02_Simulation) | Mean-field model, leaky integrate-and-fire (LIF) networks with short-term synaptic depression (STD), AIS-augmented LIF network, and a Hodgkin–Huxley-based (SAN) network | Fig. 4, Extended Data Figs. 5–10 |
| [`03_EEG_analysis/`](03_EEG_analysis) | EEG/EMG spectral analysis, Process S fitting, NREM similarity and theta-dominated wakefulness | Figs. 2 and 5, Extended Data Figs. 2, 3 and 11–13 |



## Requirements

All code was developed with **Python 3.10**. Statistical tests reported in the paper were performed in **R 4.3.1**.




## EM_analysis

Excitatory synapses in TEM images of the prelimbic cortex were analyzed in three steps: annotation, instance segmentation, and morphometry. Two Detectron2 models were trained: a two-class model for presynaptic terminals (class 0) and AZs (class 1), and a single-class model for synaptic vesicles.

### 1. Image and annotation preparation — `EM_image_prep_SD.ipynb`, `EM_image_prep_snap25.ipynb`

- Crops the information band at the bottom of the raw TEM TIFF images and generates empty [Labelme](https://github.com/wkentaro/labelme) JSON files for annotation.
- Cleans Labelme annotations (removes polygons with fewer than three vertices, unifies label names, fixes `imagePath`).
- Splits and merges annotation sets by label: `synapse` (presynaptic terminal) and `az` for the terminal/AZ model, `vesicle` for the vesicle model.
- Merges manually curated predictions back into the labeled dataset.

The two notebooks contain the same procedures for the sleep-deprivation (SD vs. control) and SNAP25 (WT vs. K40E) datasets. Annotations are used in COCO format (a `JPEGImages/` folder plus `annotations.json` per dataset).

### 2. Segmentation models — `detectron2_model_synapse.ipynb`, `detectron2_model_vesicle.ipynb`

- **Pre-processing:** images and COCO annotations are down-scaled by 0.5 (to 2000 × 1313 px); each image is normalized by its median and median absolute deviation (MAD) and saved as an 8-bit image.
- **Model:** Cascade Mask R-CNN with a ResNeXt-152 (32×8d) FPN backbone and deformable convolutions, ImageNet-5k pre-trained (`Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv`), trained with Detectron2's `DefaultTrainer`, one image per batch, frozen BN in the backbone/FPN, GN in the ROI heads and a warm-up cosine learning-rate schedule.
- **Hyperparameter search:** grid search over base learning rate and maximum iterations, then over RPN NMS, ROI-head NMS and score thresholds, using the pixel-wise F1 score (Dice) on the 16 test images as the selection criterion.
- **Evaluation:** pixel-wise F1, sensitivity and specificity, and COCO AP (`COCOEvaluator`, bbox and segm) after a custom mask-level NMS (`MaskNMSModelWrapper`).
- **Inference and curation:** the best model is applied to the unannotated images; predicted masks are converted to Labelme polygons for manual curation.

Selected hyperparameters (Extended Data Table 1):

| Parameter | Presynaptic terminal + AZ | Vesicle |
|---|---|---|
| `SOLVER.BASE_LR` | 0.0024 | 0.007 |
| `SOLVER.MAX_ITER` | 70000 | 50000 |
| `MODEL.ROI_HEADS.SCORE_THRESH_TEST` | 0.1 | 0.1 |
| `MODEL.RPN.NMS_THRESH` | 0.5 | 0.9 |
| `MODEL.ROI_HEADS.NMS_THRESH_TEST` | 0.3 | 0.5 |
| Mask-level NMS threshold (`mask_th`) | 0.01 | 80 % |

Training requires a CUDA-capable GPU. Trained weights (`model_final.pth`) are written to the Detectron2 output directory set in the notebook.

### 3. Morphometry — `detectron2_analysis_SD.ipynb`, `detectron2_analysis_snap25.ipynb`

For each curated image:

1. Vesicles are assigned to a presynaptic terminal when the vesicle-mask centroid lies inside the terminal mask.
2. AZs within `distance_threshold` = 5 px of a terminal are assigned to it; overlapping AZ masks are grouped (union–find) and the one closest to the terminal is kept.
3. Terminals with more than `vecnum_in_syn` = 10 vesicles are analyzed.
4. Per synapse, the notebook computes the number of vesicles, vesicle areas, nearest-neighbor distances between vesicles, vesicle-to-AZ distances and AZ length (major-axis length). Pixel values are converted to nm with `scale` (nm/pixel).

Per-synapse and per-vesicle values are exported as CSV/JSON (`SD-vs-nonSD/`, `WT-vs-K40E/`, and per-animal folders). The number of vesicles within 130 nm of the AZ is derived from the vesicle-to-AZ distances. Group comparisons were made at the synapse level (Mann–Whitney U test).

---

## Simulation

### Model overview

All network models use Tsodyks–Markram-type short-term synaptic depression: each presynaptic spike releases a fraction **U** (release probability) of the available vesicle resource **r**, which recovers with time constant **τ<sub>rec</sub>**; the released amount is scaled by the postsynaptic weight **w**. 

| Model | Description | Code |
|---|---|---|
| Mean-field | Three-variable (s, r, f) population model; static transfer function from the Siegert formula for a noisy LIF neuron; 4th-order Runge–Kutta, 15 s, dt = 1 ms | `mean-field.ipynb`|
| LIF network (E only) | 80 excitatory conductance-based LIF neurons, Erdős–Rényi connectivity (p = 0.2), external Poisson input and Ornstein–Uhlenbeck background conductances; forward Euler, dt = 0.1 ms, 11.5 s (analysis window 1.5–11.5 s) | `random_search_sync_to_desync_Eonly.py` and the analysis scripts/notebooks below |
| LIF network (E–I) | 64 excitatory + 16 inhibitory LIF neurons (p<sub>EE</sub> = 0.2, p<sub>EI</sub> = 0.4, p<sub>IE</sub> = 0.5, p<sub>II</sub> = 0.5), STD at both excitatory and inhibitory synapses | `random_search_sync_to_desync.py` |
| AIS-augmented LIF | E-only LIF network with an exponential integrate-and-fire (EIF) term scaled by the AIS spike-initiation excitability α<sub>AIS</sub> | `random_search_AIS_EIF_STD_sync_to_desync.py`|
| Hodgkin–Huxley-based network | 64 E + 16 I simplified averaged-neuron (SAN) model neurons (leak, delayed-rectifier K⁺, Ca²⁺, Ca²⁺-dependent K⁺, persistent Na⁺) with AMPAR/GABAR synapses, vesicle-resource STD and OU noise | `anmodel/`, `an_sws_san_net_analysis.ipynb` |

Population activity is summarized with the coefficient of variation (CV) of the population spike count in 50-ms windows (compared with the Poisson expectation CV<sub>0</sub>), the mean zero-lag pairwise Pearson correlation of 5-ms binned spike trains, population-burst statistics and the maximum firing rate in 100-ms sliding windows.


## EEG_analysis

*Code will be added.* This folder will contain the EEG/EMG analyses described in the Methods:

- vigilance-state scoring in 8-s epochs with [FASTER2](https://doi.org/10.1111/ejn.16465) (Yamada et al., 2024);
- power spectral density estimation (Welch's method; eight 50 %-overlapping Hamming-windowed sub-windows per epoch), relative NREM delta power (SWA, 0.5–4.0 Hz) and wake-to-sleep transition probability (P<sub>WS</sub>);
- fitting of the Process S model to NREM SWA dynamics (accumulation and dissipation time constants τ<sub>i</sub> and τ<sub>d</sub>);
- **NREM similarity**: cosine similarity between the normalized, log-transformed and Z-scored spectrum of each wake epoch and a reference NREM feature vector built from WT baseline recordings;
- theta-dominated wakefulness (TDW) classification.

---



## Citation

If you use this code, please cite the paper above (citation details will be updated upon publication).

## Authors

- **Fukuaki L Kinoshita**
- **Keita Yamamoto**

Department of Systems Pharmacology, Graduate School of Medicine, The University of Tokyo.
For questions about the code, please open an [issue](https://github.com/OrganismalSystemsBiology/Synapse-Sleep-Analysis/issues).



## Acknowledgements

- [Detectron2](https://github.com/facebookresearch/detectron2) (Wu et al., 2019) and Cascade R-CNN (Cai & Vasconcelos, 2021)
- [Labelme](https://github.com/wkentaro/labelme) (Wada, 2016)
- Averaged-neuron (AN/SAN) model (Tatsuki et al., *Neuron* 2016; Yoshida et al., *PNAS* 2018; [ModelDB 264519](https://modeldb.science/264519))
