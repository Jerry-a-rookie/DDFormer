# DDFormer: Dual-Domain Transformer with Flexible Channel Dependence Strategy for Time Series Anomaly Detection

[![Paper](https://img.shields.io/badge/Paper-Elsevier-blue)](URL_TO_YOUR_PAPER)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)

This is the official implementation of **DDFormer**, a Transformer-based model designed for Multivariate Time Series Anomaly Detection (MTSAD). [cite_start]DDFormer addresses the challenges of inter-channel correlations and multi-scale temporal patterns to prevent system failures in real-world applications[cite: 11, 14].

---

## 🌟 Research Highlights

* [cite_start]**Flexible Channel Dependence (FCD) Strategy**: Mitigates "cross-channel contamination" by establishing selective and directed interactions[cite: 3, 12, 46].
* [cite_start]**Dual-Domain Fusion**: Measures channel correlations using both time-domain and frequency-domain information[cite: 4, 15].
* [cite_start]**Multi-Scale Temporal Modeling**: Captures dependencies through Multi-scale Patching and a Global-Local attention mechanism[cite: 5, 17].
* [cite_start]**Superior Performance**: Achieves an average improvement of **7.12% in F1-score** over state-of-the-art methods[cite: 6, 19].

---

## 🏗️ Model Architecture

[cite_start]DDFormer jointly models temporal and channel dimensions through six core stages[cite: 190, 199]:
1.  [cite_start]**Input Normalization**: Standardizes the multivariate sequences[cite: 191].
2.  [cite_start]**FTCD Layer**: Computes asymmetric interaction info using parallel frequency and time branches[cite: 192].
3.  [cite_start]**Channel Mask Encoding**: Refines correlations using an attention-based encoder[cite: 193].
4.  [cite_start]**Multi-Scale Patching**: Covers diverse temporal resolutions via multi-granularity patch extraction[cite: 104, 194].
5.  [cite_start]**Global-Local Attention**: Captures dependencies across different temporal spans (local details vs. global trends)[cite: 104, 195].
6.  [cite_start]**Contrastive Learning**: Computes similarity between global and local representations to derive the anomaly score[cite: 196].

![DDFormer Architecture](figures/Figure_3.jpg)
[cite_start]*Figure 1: Overall framework of DDFormer, including the Channel Mask Encoder and Global-Local Attention modules[cite: 240, 241].*

### 🔬 The Channel Mask Encoder (CME)
[cite_start]The CME generates a sparse directed channel attention map to filter out noise while retaining beneficial inter-channel dependencies[cite: 269, 290]. [cite_start]It utilizes the **Gumbel-Softmax** reparameterization trick to ensure the mask matrix is differentiable[cite: 293].

![CME Algorithm](figures/image_8dab4c.png)
[cite_start]*Algorithm 1: Workflow of the Channel Mask Encoder[cite: 320].*

---

## 📊 Performance Comparison

[cite_start]DDFormer consistently outperforms baselines such as Anomaly Transformer, TimesNet, and MambaAD across five benchmark datasets: **MSL, NIPS-TS-GECCO, SMAP, SMD, and SWAT**[cite: 435, 492].

![Metric Comparison](figures/duibi.jpg)
[cite_start]*Figure 2: Model performance comparison based on Precision, Recall, and F1-score[cite: 531].*

| Metric | [cite_start]DDFormer Average Improvement [cite: 493] |
| :--- | :--- |
| **Precision** | +10.7% |
| **Recall** | +5.23% |
| **F1-score** | +5.13% |

---

## 🚀 Getting Started

### Environment Requirements
* [cite_start]**Hardware**: One NVIDIA V100 GPU (32GB), 10 vCPU cores, 64GB RAM[cite: 484].
* [cite_start]**Software**: Python 3.12, PyTorch 2.5.1, CUDA 12.4[cite: 485].

### Installation
```bash
git clone [https://github.com/YourUsername/DDFormer.git](https://github.com/YourUsername/DDFormer.git)
cd DDFormer
pip install -r requirements.txt
