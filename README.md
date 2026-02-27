# MORFI: Multimodal Zero-Shot Reasoning for Financial Time-Series Inference

**ICCV Workshop 2025** | [Paper PDF](paper/MORFI_ICCVW2025.pdf)

MORFI is a training-free framework that uses Vision-Language Models (VLMs) to predict stock prices by combining line chart images with numerical time-series data. It consistently outperforms text-only LLMs by 40–80% in MSE.

## Quick Start

### 1. Setup

```bash
git clone https://github.com/prazmara/MORFI.git
cd MORFI

# For LLM-only experiments (DeepSeek-R1)
pip install -r requirements_llm.txt

# For VLM experiments (DeepSeek-VL2)
pip install -r requirements_vlm.txt
```

### 2. Download Data

```bash
pip install yfinance
python data/download_cac40.py
```

This creates `data/preprocessed_CAC40.csv` with daily closing prices for Accor, BNP Paribas, Capgemini, and Air Liquide (2014–2020).

### 3. Run Experiments

```bash
# Text-only baseline (DeepSeek-R1-Distill-Qwen-1.5B)
python scripts/llm_text_only.py

# Multimodal without CoT (DeepSeek-VL2-Tiny)
python scripts/vlm_multimodal.py

# Multimodal with Chain-of-Thought (DeepSeek-VL2-Tiny)
python scripts/vlm_multimodal_cot.py
```

## Repo Structure

```
MORFI/
├── scripts/
│   ├── llm_text_only.py          # Text-only LLM inference
│   ├── vlm_multimodal.py         # VLM inference (image + text)
│   └── vlm_multimodal_cot.py     # VLM inference with CoT prompting
├── data/
│   ├── download_cac40.py          # Dataset download script
│   └── README.md                  # Dataset format details
├── logs/                          # Example run outputs
├── paper/                         # Published paper & supplemental
├── requirements_llm.txt           # Dependencies for LLM scripts
├── requirements_vlm.txt           # Dependencies for VLM scripts
└── README.md
```

## Results (BNP Paribas, 100-day input → 5-day forecast)

| Method | MSE |
|---|---|
| DeepSeek-R1 (text-only) | 0.035 |
| DeepSeek-VL2 (multimodal) | 0.010 |
| DeepSeek-VL2 (multimodal + CoT) | **0.005** |

Full results across all model pairs and stocks are in the [paper](paper/MORFI_ICCVW2025.pdf).

## Citation

```bibtex
@inproceedings{khezresmaeilzadeh2025morfi,
  title={MORFI: Multimodal Zero-Shot Reasoning for Financial Time-Series Inference},
  author={Khezresmaeilzadeh, Tina and Razmara, Parsa and Sadeghi, Mohammad Erfan and Azizi, Seyedarmin and Baghaei Potraghloo, Erfan},
  booktitle={ICCV Workshops},
  year={2025}
}
```

## License

MIT
