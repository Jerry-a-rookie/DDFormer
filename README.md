# DDFormer: Dual-Domain Transformer with Flexible Channel Dependence Strategy for Time Series Anomaly Detection

[![Paper](https://img.shields.io/badge/Paper-Elsevier-blue)](INSERT_YOUR_PAPER_URL_HERE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[cite_start]This repository contains the official PyTorch implementation of the paper: **"Dual-Domain Transformer with Flexible Channel Dependence Strategy for Time Series Anomaly Detection"**[cite: 7, 8]. [cite_start]DDFormer effectively addresses "cross-channel contamination" and multi-scale temporal modeling through a dual-domain approach, preventing system failures in real-world applications[cite: 11, 12].

---

## 📖 Table of Contents
- [🌟 Research Highlights](#-research-highlights)
- [🏗️ Architecture \& Methodology](#️-architecture--methodology)
- [📊 Performance](#-performance)
- [🚀 Getting Started](#-getting-started)
- [⚠️ Troubleshooting \& Version Issues](#️-troubleshooting--version-issues)
- [🖋️ Citation](#️-citation)
- [🙏 Acknowledgements](#-acknowledgements)

---

## 🌟 Research Highlights

* [cite_start]**Flexible Channel Dependence (FCD) Strategy**: Mitigates cross-channel contamination by establishing selective and directed channel interactions[cite: 47, 48].
* [cite_start]**Dual-Domain Fusion**: Measures inter-channel correlations by adaptively fusing time-domain and frequency-domain information[cite: 52, 53].
* [cite_start]**Comprehensive Multi-Scale Modeling**: Employs multi-scale patching and a Global-Local attention mechanism to capture dependencies across different temporal resolutions and spans[cite: 89, 90].
* [cite_start]**State-of-the-Art Performance**: DDFormer achieves an average improvement of **7.12% in F1-score** over existing SOTA methods across five benchmark datasets[cite: 19].

---

## 🏗️ Architecture & Methodology

[cite_start]DDFormer jointly models temporal and channel dimensions through six core stages: Input Normalization, Frequency & Time Channel Distance Layer (FTCD), Channel Mask Encoding, Multi-Scale Patching, Global-Local Attention, and Contrastive Learning [cite: 190-196].

![DDFormer Architecture](figures/Figure_3.png)
*Figure 1: Overall framework of DDFormer, detailing the Channel Mask Encoder and Global-Local Attention modules.*

### Channel Mask Encoder (CME)
[cite_start]The CME utilizes a **Gumbel-Softmax** reparameterization trick to generate a differentiable, sparse, and directed binary mask [cite: 290-293]. This ensures that only relevant channels provide information guidance, preserving the robustness of channel independence where necessary.

![CME Algorithm](figures/image_8dab4c.png)
*Algorithm 1: Workflow of the Channel Mask Encoder.*

---

## 📊 Performance

[cite_start]DDFormer was rigorously evaluated on five real-world datasets: **MSL, NIPS-TS-GECCO, SMAP, SMD, and SWAT** [cite: 442-447]. 

![Results](figures/duibi.png)
*Figure 2: Performance comparison highlighting DDFormer's superiority in Precision, Recall, and F1-score.*

[cite_start]Compared to baseline averages, DDFormer achieves remarkable improvements[cite: 493]:
* **Precision**: +10.7%
* **Recall**: +5.23%
* **F1-score**: +5.13%

---

## 🚀 Getting Started

### Environment Requirements
* **Python**: 3.12
* **PyTorch**: 2.5.1
* **CUDA**: 12.4
* [cite_start]**Hardware**: One NVIDIA V100 GPU (32GB memory) is recommended[cite: 484].

### Installation
```bash
# Clone the repository
git clone [https://github.com/YourUsername/DDFormer.git](https://github.com/YourUsername/DDFormer.git)
cd DDFormer

# Create a folder for figures and add your images (Figure_3.png, image_8dab4c.png, duibi.png)
mkdir figures

# Install dependencies
pip install -r requirements.txt

#⚠️ Troubleshooting & Version Issues
##1. NumPy 2.0 Compatibility Error
AttributeError: `np.Inf` was removed in the NumPy 2.0 release. Use `np.inf` instead.

Solution: This is due to the removal of the uppercase alias in NumPy 2.0. To fix this, locate the line in solver.py (typically around line 34) and change np.Inf to the lowercase version:

##2.CUDA Initialization & Driver Version
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12080).
Solution: Your current NVIDIA driver version does not support the latest PyTorch/CUDA 12.4 build. Please either:

Update your NVIDIA GPU driver to the latest version.

Install a specific PyTorch version compiled for your existing older driver (visit pytorch.org for previous version installation commands).
