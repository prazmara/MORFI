# MORFI: Multimodal Zero-Shot Reasoning for Financial Time-Series Inference

**ICCV Workshop 2025** | [Paper](https://openaccess.thecvf.com/content/ICCV2025W/MMFM/papers/Khezresmaeilzadeh_MORFI_Mutimodal_Zero-Shot_Reasoning_for_Financial_Time-Series_Inference_ICCVW_2025_paper.pdf) | [Supplementary](https://openaccess.thecvf.com/content/ICCV2025W/MMFM/supplemental/Khezresmaeilzadeh_MORFI_Mutimodal_Zero-Shot_ICCVW_2025_supplemental.pdf)

MORFI is a training-free framework that uses Vision-Language Models (VLMs) to predict stock prices by combining line chart images with numerical time-series data. It outperforms text-only LLMs by 40–80% in MSE.

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/prazmara/MORFI.git
cd MORFI
```

### 2. Install Dependencies

```bash
# For LLM-only experiments (DeepSeek-R1)
pip install -r requirements_llm.txt

# For VLM experiments (DeepSeek-VL2)
pip install -r requirements_vlm.txt
```

### 3. Download Data

```bash
pip install yfinance
python data/download_cac40.py
```

### 4. Run

```bash
# Text-only baseline (DeepSeek-R1-Distill-Qwen-1.5B)
python scripts/llm_text_only.py

# Multimodal without CoT (DeepSeek-VL2-Tiny)
python scripts/vlm_multimodal.py

# Multimodal with Chain-of-Thought (DeepSeek-VL2-Tiny)
python scripts/vlm_multimodal_cot.py
```

---

## Repo Structure

```
MORFI/
├── scripts/
│   ├── llm_text_only.py            # Text-only LLM inference
│   ├── vlm_multimodal.py           # VLM inference (image + text)
│   └── vlm_multimodal_cot.py       # VLM with CoT prompting
├── data/
│   ├── download_cac40.py           # Dataset download script
│   └── README.md                   # Dataset details
├── requirements_llm.txt
├── requirements_vlm.txt
├── .gitignore
└── README.md
```

## Results (BNP Paribas, 100-day input → 5-day forecast)

| Method | MSE |
|---|---|
| DeepSeek-R1 (text-only) | 0.035 |
| DeepSeek-VL2 (multimodal) | 0.010 |
| DeepSeek-VL2 (multimodal + CoT) | **0.005** |

Full results across all model pairs and four stocks are in the [paper](https://openaccess.thecvf.com/content/ICCV2025W/papers/Khezresmaeilzadeh_MORFI_Mutimodal_Zero-Shot_Reasoning_for_Financial_Time-Series_Inference_ICCVW_2025_paper.pdf).

## Citation

```bibtex
@InProceedings{Khezresmaeilzadeh_2025_ICCV,
    author    = {Khezresmaeilzadeh, Tina and Razmara, Parsa and Sadeghi, Mohammad Erfan and Azizi, Seyedarmin and Potraghloo, Erfan Baghaei},
    title     = {MORFI: Mutimodal Zero-Shot Reasoning for Financial Time-Series Inference},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {4236-4245}
}
```

## License

MIT
