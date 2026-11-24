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
| [`03_NREM_similarity/`](03_NREM_similarity) | NREM similarity: cosine similarity between wake EEG spectra and a reference NREM spectral vector, its relation to wake-to-sleep transition probability, and comparisons between WT and SNARE-mutant mice | Fig. 5c–g |



## Requirements

The EM_analysis and Simulation code was developed with **Python 3.10**. The NREM_similarity scripts and the statistical tests reported in the paper use **R 4.3.1**.




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


## NREM_similarity

R scripts that compute **NREM similarity**, an EEG-based metric of how closely the cortical activity of each wake epoch resembles NREM sleep, and reproduce the analyses in Fig. 5c–g.

| Script | Dataset | Analyses | Figures |
|---|---|---|---|
| `EEG_NREM-similarity_Pws_Psw_Bmal1_without_Spearman.R` | WT and *Bmal1* KO mice, 6 days (LD 0–72 h, then DD 72–144 h) | Hourly NREM similarity versus wake-to-sleep (P<sub>WS</sub>) and sleep-to-wake (P<sub>SW</sub>) transition probabilities; individual-relative binning and Spearman's rank correlation | Fig. 5c,d |
| `EEG_NREM similarity_without_Timeseries-KDE-Bar_K40E.R` | SNAP25-WT vs. SNAP25-K40E, 3-day baseline | Time course, distribution and light/dark comparison of NREM similarity | Fig. 5e |
| `EEG_NREM similarity_without_Timeseries-KDE-Bar_4K.R` | SNAP25-WT vs. SNAP25-4K, 3-day baseline | Same as above | Fig. 5f |
| `EEG_NREM similarity_without_Timeseries-KDE-Bar_AVVA.R` | STX1A-WT vs. STX1A-AVVA, 3-day baseline | Same as above | Fig. 5g |

The three mutant-cohort scripts are identical except for the input file, group names and colors.

### Input data

Each script loads a Python pickle (`psd_info_list_<cohort>.pkl`) through `reticulate`. The pickle is a list with one entry per mouse containing:

- `mouse_id`, `mouse_group` – animal ID and genotype label (labels such as `SNAP25_WT`/`SNAP25_K40E` are renamed to `WT`/`K40E` in the script);
- `raw` – epoch × frequency matrix of EEG power spectral densities for consecutive 8-s epochs;
- `stage_call` – vigilance state of each epoch (`WAKE`, `NREM`, `REM`; scored with [FASTER2](https://doi.org/10.1111/ejn.16465));
- `freq_bins` – frequencies (Hz) of the columns of `raw`.

Recordings are assumed to start at ZT0: hours are counted from the first epoch, and hours 0–11 of each day are treated as the light phase. Before running, set `local_filepath` (path to the pickle) and `RETICULATE_PYTHON` (a Python installation with `numpy`/`pandas`) at the top of each script.

### NREM similarity (common to all scripts)

1. Each epoch's spectrum is normalized to its total power (%) and log10-transformed.
2. **Reference spectra:** all epochs from the WT mice of the same cohort during the first 3 days (LD baseline) are used to compute the per-frequency mean (μ<sub>wt</sub>) and standard deviation (σ<sub>wt</sub>); every epoch of every mouse is Z-scored with μ<sub>wt</sub> and σ<sub>wt</sub>.
3. **NREM feature vector:** V<sub>NREM_feat.</sub> is the mean Z-scored spectrum across all WT NREM epochs.
4. **NREM similarity** of each epoch is the cosine similarity between its Z-scored spectrum and V<sub>NREM_feat.</sub>.
5. **Strict wake filter:** only wake epochs whose preceding and following epochs are also wake are used.

### Circadian analysis — `EEG_NREM-similarity_Pws_Psw_Bmal1_without_Spearman.R` (Fig. 5c,d)

- **Hourly values:** mean NREM similarity of strict-wake epochs (hours with > 30 strict-wake epochs); P<sub>WS</sub> = (wake→sleep transitions / wake epochs) × 100 and P<sub>SW</sub> = (sleep→wake transitions / sleep epochs) × 100, with NREM and REM pooled as sleep (hours with > 30 wake or sleep epochs). The merged hourly table is cached as `01_Merged_Hourly_Data.csv`; delete it to recompute from the pickle.
- **Overlays:** group-mean and individual time courses of NREM similarity together with P<sub>WS</sub>, −P<sub>SW</sub> and P<sub>WS</sub> − P<sub>SW</sub>, separately for LD (0–72 h) and DD (72–144 h).
- **Individual-relative binning:** for each mouse and period, the hourly NREM-similarity values are divided into equal-width rank bins (6 bins in the paper; the script also outputs 7 and 8 bins), and the mean transition probability is computed per bin. Group means are weighted by the number of hours per bin, and the association between rank and group-mean P<sub>WS</sub> is tested with Spearman's rank correlation (the maximal information coefficient, MIC, is also reported).
- **Outputs** (in `EEG_NREM-similarity_Pws_Psw_without/`): PDF figures, plot-data CSVs, binned data per mouse (`Individual_Binned_Data/`), correlation statistics (`15_Relative_Binned_Detailed_Stats_LD_DD.csv`) and a PowerPoint summary.

### Mutant cohorts — `EEG_NREM similarity_without_Timeseries-KDE-Bar_<K40E|4K|AVVA>.R` (Fig. 5e–g)

Using the first 3 days of baseline recording:

- **Stage-wise spectra:** mean Z-scored spectra (0–30 Hz) of wake, NREM and REM epochs for each genotype, Z-scored both per animal and against the WT reference.
- **Distribution:** kernel density estimates of NREM similarity across all strict-wake epochs for each genotype.
- **Time course:** hourly mean NREM similarity (mean ± SEM across mice) over 72 h; the dotted line marks the mean NREM similarity of WT reference wake epochs.
- **Light/dark comparison:** per-mouse mean NREM similarity over 24 h, the light phase (ZT0–12) and the dark phase (ZT12–24), shown as box plots. WT and mutant groups are compared with Student's t-test, Welch's t-test or the Mann–Whitney U test depending on normality and equality of variances; the results of all candidate tests (Shapiro–Wilk, F-test, Wilcoxon, Welch, Student) are written to `Stats_Results_Detailed_Summary_Boxplots.csv`.
- **Outputs** (in `outputs/`): PDF figures, plot-data CSVs and per-mouse summaries (`Projection_Phase_Summary.csv`, `Projection_Timecourse_Summary.csv`, `Projection_AllScores_Raw.csv`).

### Requirements and usage

R 4.3.1 with `tidyverse`, `reticulate`, `car` and `ggh4x`; the *Bmal1* script additionally uses `minerva` and `officer`. Outputs are written relative to the working directory:

```bash
cd 03_NREM_similarity
Rscript "EEG_NREM similarity_without_Timeseries-KDE-Bar_K40E.R"
Rscript EEG_NREM-similarity_Pws_Psw_Bmal1_without_Spearman.R
```

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
