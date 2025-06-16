# Remaining Lifespan Prediction from Images with Uncertainty Estimation

This repository contains the code for the paper
**"Uncertainty-Aware Remaining Lifespan Prediction from Images"**.

📄 **[Original Paper (TODO: Add Link)](TODO)**

---

## 🚀 Getting Started

Clone the repository and set up your environment:

```bash
git clone https://github.com/YOUR-USERNAME/RemainingLifespanPrediction.git
cd RemainingLifespanPrediction

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Training the Model

The main training script is in `main.py`.

To train the model, simply run:

```bash
python main.py
```

**Note:**
The first time you run the script, the dataset will be downloaded from Huggingface. Depending on your internet connection, this may take some time.

When you change the dino model you also need to change the embedding dimensionality. Look at [DINOv2](https://github.com/facebookresearch/dinov2)

---

## 📦 Dataset

You can use the dataset directly with the [Huggingface `datasets`](https://huggingface.co/docs/datasets) library:

```python
from datasets import load_dataset

dataset = load_dataset("TristanKE/RemainingLifespanPredictionFaces", split="train")
```

* **Note:**
  There is **no premade train/test split**.
  By choosing `split="train"` you get the entire dataset.
  Please create your own splits (see `main.py` for an example).

---

## 📄 Citation

If you use this code or dataset, please cite our paper:

```bibtex
@article{TODO,
  title={Uncertainty-Aware Remaining Lifespan Prediction from Images},
  author={Kenneweg, Tristan and ...},
  journal={TODO},
  year={2025},
  url={TODO}
}
```

---


## 📧 Contact

For questions or collaborations, please contact
Tristan Kenneweg — \[tkenneweg@techfak.uni-bielefeld.de]

---

## License

[MIT License](LICENSE)
© Tristan Kenneweg 2025
