# Conversational Repair in Text-to-Image Generation (DiffusionDB)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

This repository contains the replication dataset, analytical code, and econometric modeling for the empirical research:

> **"Empirical Analysis of Conversational Repair in Text-to-Image Interaction Using DiffusionDB"**  
> *Athallah Pandhu Nitisara, et al.*  
> Department of Information Systems, School of Industrial Engineering, Telkom University, Bandung, Indonesia.

---

## 📌 Research Overview

Users interacting with generative text-to-image systems (such as Stable Diffusion) rarely obtain their desired visual outcome on the first attempt. Consequently, they engage in iterative, multi-turn prompt adjustments. This study conceptualizes this prompt modification cycle through the linguistic framework of **Conversational Repair** (Schegloff et al., 1977).

Using public production logs from the **DiffusionDB 2M** repository:
1. **RQ1 (Taxonomy & Distribution)**: We classify consecutive prompt revisions ($\Delta t \le 30$ min) into a 5-class operational repair taxonomy (`ADD_DETAIL`, `STYLE_MODIFIER`, `SUBSTITUTE`, `SIMPLIFY`, `REWORD`) using Google Gemini (Gemini Flash), validated by double-blind human raters ($n = 200$; $P_o \ge 79.0\%$, PABAK $0.58–0.72$).
2. **RQ2 (Multimodal Alignment Gain)**: We compute image-text representation alignment using **OpenCLIP ViT-B/32** (LAION-2B) on $N = 400$ paired interactions (800 real images) and evaluate alignment gain ($\Delta S = S_2 - S_1$) using Wilcoxon signed-rank testing.
3. **RQ3 (Econometric Modeling)**: We estimate a multivariate OLS regression model with **Cluster-Robust Standard Errors (CRSE)** clustered by `user_id` to evaluate the marginal association of each repair strategy under technical generation controls ($\Delta\text{Seed}, \Delta\text{CFG}, \Delta\text{Step}, \Delta t$).

---

## 🔬 Key Empirical Findings

* **Linguistic Saturation**: Repair strategy proportions remain virtually identical between a 1,000-pair pool and a 2,500-pair pool (`SUBSTITUTE`: 56.6%, `ADD_DETAIL`: 43.6%, `STYLE_MODIFIER`: 32.3%, `REWORD`: 17.9%, `SIMPLIFY`: 13.5%), confirming behavioral convergence. Multi-label composite editing accounts for **56.1%** of interactions.
* **Metric Blindness & Null Results (RQ2)**: Median $\Delta S = 0.0007$ ($p = 0.347$, effect size $r = 0.02$). Only **50.75%** of user revisions yield positive alignment changes, demonstrating that manual prompt hacking in the wild does not reliably improve global CLIP representation.
* **Partial Associations (RQ3)**: Regression analysis reveals no statistically significant individual lexical effect at $\alpha = 0.05$ (`ADD_DETAIL` $b = 0.0045, p = 0.505$; `STYLE_MODIFIER` $b = -0.0015, p = 0.830$; model $R^2 = 0.021$). However, `REWORD` demonstrates a marginal positive trend ($b = 0.0191, p = 0.067$). Structural robustness checks ($\Delta t \le 15$m vs. $\Delta t \le 30$m) confirm identical coefficient trajectories.
* **Interaction Design Guidelines**: Empirical findings are synthesized into three interaction design principles compliant with **ISO 9241-210** and **Google PAIR**:
  1. *Structured Semantic Input Separation* (decoupling entities, modifiers, and styles).
  2. *Automated Latent Space Anchoring / Seed Locking* (isolating semantic edits from stochastic drift).
  3. *Bidirectional Lexical Rewording Assistant* (suggesting contextual paraphrases).

---

## 📂 Repository Structure

```text
conversational-repair-diffusiondb/
├── data/
│   ├── dataset_final_multimodal_400.csv     # N=400 multimodal pair evaluation dataset
│   ├── tabel_alur_sampel.csv                # Data triage funnel (2M -> 1,336 -> 400)
│   ├── tabel_cohen_kappa_validasi.csv       # Human validation metrics (Po, Kappa, PABAK)
│   ├── tabel_regresi_crse_side_by_side.csv  # OLS CRSE regression tables (15m vs 30m)
│   ├── tabel_rq1_distribusi_taksonomi.csv   # Repair taxonomy distributions & 95% CI
│   └── tabel_vif_multikolinearitas.csv      # Variance Inflation Factor diagnostics
├── figures/
│   ├── fig1_rq1_distribusi_frekuensi.png    # Taxonomy frequency bar chart
│   ├── fig2_rq1_ko_okurensi_multilabel.png  # Multi-label co-occurrence matrix heatmap
│   ├── fig3_rq2_distribusi_delta_clip.png   # Density distribution of Delta S
│   └── fig4_rq3_forest_plot_regresi.png     # Forest plot of CRSE coefficients
├── notebooks/
│   └── conversational_repair_diffusiondb.ipynb # End-to-end reproducible pipeline
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🚀 Quickstart & Replication

### 1. Clone the repository
```bash
git clone https://github.com/pandhuuuu/conversational-repair-diffusiondb.git
cd conversational-repair-diffusiondb
```

### 2. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Analysis
You can open and execute the Jupyter Notebook directly:
```bash
jupyter notebook notebooks/conversational_repair_diffusiondb.ipynb
```
Or execute on Google Colab with GPU runtime for OpenCLIP inference.

---

## 📦 Dependencies

* Python >= 3.10
* `pandas >= 2.0.0`
* `numpy >= 1.24.0`
* `scipy >= 1.10.0`
* `statsmodels >= 0.14.0` (for OLS with cluster-robust standard errors)
* `open-clip-torch >= 2.20.0`
* `sentence-transformers >= 2.2.2`
* `matplotlib >= 3.7.0`
* `seaborn >= 0.12.0`

---

## 📜 Citation

If you use this dataset, taxonomy, or code in your academic research, please cite:

```bibtex
@article{naufal2026conversational,
  title={Empirical Analysis of Conversational Repair in Text-to-Image Interaction Using DiffusionDB},
  author={Nitisara, Athallah Pandhu and Pembimbing, Dosen},
  journal={Jurnal Nasional Teknik Elektro dan Teknologi Informasi (JNTETI)},
  year={2027},
  publisher={Universitas Gadjah Mada}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
