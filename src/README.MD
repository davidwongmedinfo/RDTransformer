# Reference Difference Transformer (RDTransformer)
**RDTransformer** is a Transformer-based model designed to predict the functional effects of mutations in biological sequences (RNA, DNA, or proteins). Built upon a Transformer backbone, it introduces a **dynamic difference-based embedding mechanism** relative to the wild-type sequence. This mechanism filters out noise from non-mutational sites while emphasizing the effects of introduced or natural mutations.

---

## 🧠 Architecture
![Architecture of the RDTransformer](https://github.com/davidwongmedinfo/RDTransformer/tree/main/architecture.png)

---

## 📚 Data
The pretraining data was sourced from the [**RNAcentral** database](https://rnacentral.org/).
**Citation:**
> RNAcentral Consortium.
> RNAcentral 2021: secondary structure integration, improved sequence search and new member databases.
> Nucleic Acids Research, 2021; 49(D1):D212-D220.
> [https://doi.org/10.1093/nar/gkaa921](https://doi.org/10.1093/nar/gkaa921)

Both raw and preprocessed pretraining datasets are available [here](https://github.com/davidwongmedinfo/RDTransformer/tree/main/data).

---

## 💻 Source Code
The full implementation is available [here](https://github.com/davidwongmedinfo/RDTransformer/tree/main/src).

---

## ⚙️ Environment Setup
**1. Create environment:**
```bash
conda env create -f rdt_env.yml
```
> **Note**: This is a CPU-only environment to maximize compatibility. The original experiments in the paper were conducted with the following configuration:
> - PyTorch 2.6.0 + CUDA 12.4  
> - NVIDIA driver 555.42.06
> - cuDNN: as bundled with the PyTorch 2.6.0+cu124 distribution
>
> To reproduce GPU-accelerated results, install the matching CUDA-enabled PyTorch wheel: 
> ```bash
> pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
> ```

**2. Activate environment:**
```bash
conda activate rdt_env
```
