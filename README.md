# Res<sup>2</sup>CLIP: Few-Shot Generalist Anomaly Detection with Residual-to-Residual Alignment


Res<sup>2</sup>CLIP is a few-shot generalist anomaly detection framework that aligns visual residuals with text residuals using a frozen CLIP backbone. It supports two modes:

| Mode | Symbol | Description |
|------|--------|-------------|
| `training-free` | Res<sup>2</sup>CLIP<sup>*</sup> | Direct three-branch fusion on frozen CLIP features without fine-tuning. |
| `finetune`      | Res<sup>2</sup>CLIP<sup>†</sup> | Lightweight adapters trained on an auxiliary dataset for higher performance. |

---

### Environment Preparation

```bash
conda create -n res2clip python=3.10
conda activate res2clip
pip install -r requirements.txt
```

---

### Dataset Preparation

Dataset metadata JSON files are generated following the same procedure as AnomalyCLIP, please refer to [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) for scripts and instructions.

---

### Backbone Preparation

We use the CLIP **ViT-L/14@336px** backbone. The model is **downloaded automatically** on first run to `./clip_model/ViT-L-14-336px.pt` (or download manually from the [OpenAI CLIP releases](https://github.com/openai/CLIP) and place it there).

---

### Training

Edit paths in `train.sh`, then:

```bash
bash train.sh
```

Adapters are trained separately on MVTec AD and VisA. Checkpoints are saved to `./checkpoints/{mvtec,visa}/`.

---

### Evaluation

**Training-free** (Res<sup>2</sup>CLIP<sup>*</sup>):

```bash
bash test_trainingfree.sh
```

**Fine-tuned** (Res<sup>2</sup>CLIP<sup>†</sup>):

```bash
bash test_finetune.sh
```


---

### Acknowledgement

We thank [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) for their open-source codebase, on which `clip_lib/` is based.

---

## Citation
If you think this work is helpful to you, please consider citing our paper.

```bibtex
@article{liu2026res2clip,
  title={Res$^2$CLIP: Few-Shot Generalist Anomaly Detection with Residual-to-Residual Alignment},
  author={Liu, Xinyue and Wang, Jianyuan and Leng, Biao and Zhang, Shuo},
  journal={arXiv preprint arXiv:2605.16171},
  year={2026}
}
```
