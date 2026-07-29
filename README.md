# Contact Matrix: Enhancing Dance Motion Synthesis with Precise Interaction Modeling (CVPR 2026 Findings)

This repository is the official PyTorch implementation of Contact Matrix: Enhancing Dance Motion Synthesis with Precise Interaction Modeling ([paper](https://openaccess.thecvf.com/content/CVPR2026F/html/Chen_Contact_Matrix_Enhancing_Dance_Motion_Synthesis_with_Precise_Interaction_Modeling_CVPRF_2026_paper.html)). Please feel free to contact us if you have any questions.

## Requirements

Create a Python environment and install PyTorch following your CUDA version. The reported experiments were conducted with PyTorch 1.13 and CUDA 11.7.

```bash
conda create -n rcdiff python=3.10 -y
conda activate rcdiff
pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

Install EasyVolcap from the commit used in our experiments:

```bash
git clone https://github.com/zju3dv/EasyVolcap.git
cd EasyVolcap
git checkout 1175727f
pip install -e .
cd ..
```

Install this repository:

```bash
git clone https://github.com/ByChelsea/rcdiff.git
cd rcdiff
pip install -r requirements.txt
pip install -e .
```

For contact-frequency evaluation, install the SMPL-X dependency and compile the mesh collision extension:

```bash
pip install smplx trimesh
cd rcdiff/utils/motion_metrics/contact/torch_mesh_isect
python setup.py install
cd ../../../../..
```

## Data

We use the DD100 dataset released by [Duolando](https://lisiyao21.github.io/projects/Duolando/). Download the dataset from their project page and update the data paths in `configs/exps/*.yaml`.

## Pretrained Models

Download pretrained models from [Hugging Face](https://huggingface.co/ByChelsea123/RCDiff):

```bash
pip install -U huggingface_hub
hf download ByChelsea123/RCDiff --repo-type model --local-dir .
```

Place pretrained checkpoints and record files under `data/trained_model` and `data/record`:

```text
data/trained_model/
  partfusion-vq/
  transl-vq/
  contact-vq/
  rcdiff/
data/record/
  partfusion-vq/
  transl-vq/
  contact-vq/
```

The translation VQ follows the same design as Duolando, so we use their pretrained translation VQ model directly. The RCDiff config points to the VQ config records through:

```yaml
motoken_cfg_file: data/record/partfusion-vq/xxx.yaml
transl_cfg_file: data/record/transl-vq/xxx.yaml
contact_cfg_file: data/record/contact-vq/xxx.yaml
```

## Training

### VQ Models

Train the low-level VQ models:

```bash
python rcdiff/scripts/main.py -t train -c configs/exps/partfusion-vq.yaml
python rcdiff/scripts/main.py -t train -c configs/exps/contact-vq.yaml
```

The translation VQ can be trained with:

```bash
python rcdiff/scripts/main.py -t train -c configs/exps/transl-vq.yaml
```

### RCDiff Normalization

RCDiff normalizes VQ latents and music features using train-set statistics:

```bash
python rcdiff/scripts/compute_rcdiff_norm.py -c configs/exps/rcdiff.yaml --out npys
```

### RCDiff

```bash
python rcdiff/scripts/main.py -t train -c configs/exps/rcdiff.yaml
```

EMA is enabled for RCDiff in the config. During validation/testing, the runner applies EMA weights automatically when the EMA checkpoint exists.

## Testing and Evaluation

### VQ Models

VQ reconstruction metrics are computed by running test:

```bash
python rcdiff/scripts/main.py -t test -c configs/exps/partfusion-vq.yaml
python rcdiff/scripts/main.py -t test -c configs/exps/transl-vq.yaml
python rcdiff/scripts/main.py -t test -c configs/exps/contact-vq.yaml
```

### RCDiff

Generate test motions with EMA weights:

```bash
python rcdiff/scripts/main.py -t test -c configs/exps/rcdiff.yaml
```

To test raw weights instead of EMA:

```bash
python rcdiff/scripts/main.py -t test -c configs/exps/rcdiff.yaml runner_cfg.ema_cfg.enabled=False
```

### RCDiff Metrics

Install the SMPL-X model and compile the mesh collision extension if you need CF:

```bash
cd rcdiff/utils/motion_metrics/contact/torch_mesh_isect
python setup.py install
```

Run metrics for generated motions, for example epoch 300:

```bash
python rcdiff/scripts/evaluate_rcdiff_metrics.py \
  --pred-root /path/to/rcdiff/eval/300 \
  --gt-root /path/to/DD100/motion/pos3d/all \
  --music-root /path/to/DD100/music/feature/all \
  --smplx-model-root /path/to/smplx_models \
  --json-out /path/to/rcdiff/eval_metrics_300.json
```

Contact-frequency evaluation is time-consuming. Add `--skip-cf` to skip it during quick evaluation.

Multiple generated result folders can be evaluated in one command:

```bash
python rcdiff/scripts/evaluate_rcdiff_metrics.py \
  --pred-root \
    /path/to/rcdiff/eval/<epoch-a> \
    /path/to/rcdiff/eval/<epoch-b> \
    /path/to/rcdiff/eval/<epoch-c> \
  --gt-root /path/to/DD100/motion/pos3d/all \
  --music-root /path/to/DD100/music/feature/all \
  --smplx-model-root /path/to/smplx_models \
  --json-out /path/to/rcdiff/eval_metrics.json
```

GT metric features are cached under `data/metric_cache` and reused automatically. Use `--refresh-gt-cache` to recompute them.

## Citation

If you find this work useful, please consider citing:

```bibtex
@InProceedings{rcdiff,
  author    = {Chen, Xuhai and Cen, Zhi and Pi, Huaijin and Peng, Sida and Zhou, Xiaowei and Liu, Yong},
  title     = {Contact Matrix: Enhancing Dance Motion Synthesis with Precise Interaction Modeling},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  month     = {June},
  year      = {2026}
}
```

## Acknowledgement

This codebase is built upon [EasyVolcap](https://github.com/zju3dv/EasyVolcap). We also thank [Duolando](https://lisiyao21.github.io/projects/Duolando/) for their excellent work and for releasing the DD100 dataset.
