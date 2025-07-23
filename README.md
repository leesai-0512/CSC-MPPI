# CSC-MPPI: A Novel Constrained MPPI Framework with DBSCAN for Reliable Obstacle Avoidance

This repository provides a **JAX-JIT accelerated implementation** of the **Constrained Sampling Cluster MPPI (CSC-MPPI)** algorithm.  
CSC-MPPI introduces a novel constrained sampling framework based on **DBSCAN clustering** and **primal-dual gradient optimization** to improve obstacle avoidance and constraint satisfaction in sampling-based Model Predictive Path Integral (MPPI) control.

The algorithm is designed for high-performance execution on GPU using **JAX with just-in-time (JIT) compilation**, enabling fast and parallelized trajectory rollouts.

---

## 📰 Publication

**📄 Title:** *CSC-MPPI: A Novel Constrained MPPI Framework with DBSCAN for Reliable Obstacle Avoidance*  
**🛠 Authors:** Leesai Park¹, Keunwoo Jang²†, and Sanghyun Kim¹³†  
**📅 Conference:** IEEE/RSJ International Conference on Intelligent Robots and Systems (**IROS**) 2025  

---

## ⚙️ Prerequisites

- Python 3.10+
- JAX with GPU support
- CUDA 12.1+
- Recommended: Use a virtual environment (e.g., conda)

### ✅ Create Virtual Environment
Please refer to the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for detailed setup instructions compatible with your system and CUDA version.

```bash
conda create -n csc_mppi python=3.10
conda activate csc_mppi

conda install matplotlib
pip install -U "jax[cuda12]"
conda install -c rapidsai -c nvidia -c conda-forge -c defaults  cuml

```

---

## 🚀 Installation and Run

```bash
git clone <repository_url>
cd <repository_name>
python3 main/env1.py or main/env2.py
```

---

## 🎬 Experiment Results

### Environment 1

| CSC-MPPI | Standard MPPI |
|---------------|----------|
| ![](gifs/env1_csc-mppi.gif) | ![](gifs/env1_standard_mppi.gif) |

---

### Environment 2

| CSC-MPPI | CSC-MPPI wo DBSCAN |
|---------------|----------|
| ![](gifs/env2_csc-mppi.gif) | ![](gifs/env2_csc-mppi_wo_dbscan.gif) |

---
