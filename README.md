## Navigation World Models, CVPR 2025 (Oral) <br><sub>Official PyTorch Implementation</sub>

> NavWare fork: LT-NWM research status, reproducible setup, training/inference handoff, and experiment gates are documented in [docs/README.md](docs/README.md). The hybrid memory branch is still under validation; read its status before launching cluster jobs.

### [Paper](https://arxiv.org/abs/2412.03572) | [Project Page](https://www.amirbar.net/nwm/) | [Notebook Demo](interactive_model.ipynb) | [Models](https://huggingface.co/facebook/nwm)

This repo contains the official PyTorch implementation of Navigation World Models- the Conditional Diffusion Transformer (CDiT) model training code. See the [project page](https://www.amirbar.net/nwm) for additional results.

> [**Navigation World Models**](https://www.amirbar.net/nwm)<br>
> [Amir Bar](https://www.amirbar.net), [Gaoyue "Kathy" Zhou](https://gaoyuezhou.github.io/), [Danny Tran](https://dannytran123.github.io/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Yann LeCun](http://yann.lecun.com/)
> <br>AI at Meta, UC Berkeley, New York University<br>

## Setup
First, download and set up the repo:

```bash
git clone https://github.com/facebookresearch/nwm
cd nwm
```

## Data
To download and preprocess data, please follow the steps from [NoMaD](https://github.com/robodhruv/visualnav-transformer?tab=readme-ov-file#data-wrangling), specifically:
- Download the datasets
- Change the [preprocessing resolution](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/data/data_utils.py#L13) from (160, 120) to (320, 240) for higher resolution 
- run `process_bags.py` and `process_recon.py` to save each processed dataset to `path/to/nwm_repo/data/<dataset_name>`.

For [SACSon/HuRoN](https://sites.google.com/view/sacson-review/huron-dataset), we used a private version which contains higher resolution images. Please contact the authors for access.

Finally, you should have the following structure:

```
nwm/data
├── <dataset_name>
│   ├── <name_of_traj1>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_1.jpg
│   │   └── traj_data.pkl
│   ├── <name_of_traj2>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_2.jpg
│   │   └── traj_data.pkl
│   ...
└── └── <name_of_trajN>
    	├── 0.jpg
    	├── 1.jpg
    	├── ...
        ├── T_N.jpg
        └── traj_data.pkl
```  


## Requirements

The reproducible setup script selects the PyTorch backend explicitly and
installs the pinned project dependency groups. For WSL with an NVIDIA GPU:

```bash
sudo apt update
sudo apt install -y python3.10-venv ffmpeg

mkdir -p ~/.venvs
bash setup_nwm_env.sh \
  --profile nwm \
  --backend cu124 \
  --venv ~/.venvs/navware-nwm

ln -s ~/.venvs/navware-nwm .venv-wsl
source .venv-wsl/bin/activate
python scripts/navware.py doctor --profile nwm
```

See [`docs/10_setup/LOCAL_WSL.md`](docs/10_setup/LOCAL_WSL.md) for disk
layout, CPU-only setup, validation, and troubleshooting.

## Training

Using torchrun:
```bash
export NUM_NODES=8
export HOST_NODE_ADDR=<HOST_ADDR>
export CURR_NODE_RANK=<NODE_RANK>

torchrun \
  --nnodes=${NUM_NODES} \
  --nproc-per-node=8 \
  --node-rank=${CURR_NODE_RANK} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${HOST_NODE_ADDR}:29500 \
  train.py --config config/nwm_cdit_xl.yaml --ckpt-every 2000 --eval-every 10000 --bfloat16 1 --epochs 300 --torch-compile 0
```

Or using submitit and slurm (8 machines of 8 gpus):
```bash
python submitit_train_cw.py --nodes 8 --partition <partition_name> --qos <qos> --config config/nwm_cdit_xl.yaml --ckpt-every 2000 --eval-every 10000 --bfloat16 1 --epochs 300  --torch-compile 0
```

Or locally on one GPU for debug:
```bash
python train.py --config config/nwm_cdit_xl.yaml --ckpt-every 2000 --eval-every 10000 --bfloat16 1 --epochs 300  --torch-compile 0
```

Note: torch compile can lead to ~40%  faster training speed. However, it might lead to instabilities and inconsistent behvaior across different pytorch versions. Use carefuly.

## Pretrained Models
To use a pretrained CDiT/XL model:
- Download a pretrained model from [Hugging Face](https://huggingface.co/facebook/nwm)
- Place the checkpoint in ./logs/nwm_cdit_xl/checkpoints

# Evaluation

directory to save evaluation results:
`export RESULTS_FOLDER=/path/to/res_folder/`

## Evaluate on single time step prediction 

### 1. Prepare ground truth frames for evaluation (one-time)

```bash
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --datasets recon,scand,sacson,tartan_drive \
    --batch_size 96 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1
```
### 2. Predict future state given action

```bash    
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp 0100000 \
    --datasets <dataset_name> \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}
```
### 3. Report metrics compared to GT (LPIPS, DreamSim, FID)

```bash    
python isolated_nwm_eval.py \
    --datasets <dataset_name> \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl \
    --eval_types time
```
Results are saved in ${RESULTS_FOLDER}/nwm_cdit_xl/<dataset_name>

## Evaluate on following ground truth trajectories

### 1. Prepare ground truth frames for evaluation (one-time)

```bash
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --datasets recon,scand,sacson,tartan_drive \
    --batch_size 96 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1 \
    --rollout_fps_values 1,4
```
### 2. Simulate a GT trajectory using NWM
```bash
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp 0100000 \
    --datasets <dataset_name> \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER} \
    --rollout_fps_values 1,4
```

### 3. Report metrics compared to GT trajectories (LPIPS, DreamSim, FID)
```bash
    python isolated_nwm_eval.py \
        --datasets recon \
        --gt_dir ${RESULTS_FOLDER}/gt \
        --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl \
        --eval_types rollout
```
Results are saved in ${RESULTS_FOLDER}/nwm_cdit_xl/<dataset_name>

### Trajectory Evaluation - Planning

Using 1-step Cross Entropy Method planning on 8 gpus (sampling 120 trajectories):
```bash
torchrun --nproc-per-node=8 planning_eval.py \
    --exp config/nwm_cdit_xl.yaml   \
    --datasets recon   \
    --rollout_stride 1   \
    --batch_size 1   \
    --num_samples 120   \
    --topk 5   \
    --num_workers 12   \
    --output_dir ${RESULTS_FOLDER}   \
    --save_preds   \
    --ckp 0100000   \
    --opt_steps 1   \
    --num_repeat_eval 3
```
Results are saved in ${RESULTS_FOLDER}/nwm_cdit_xl/<dataset_name>

## BibTeX

```bibtex
@article{bar2024navigation,
  title={Navigation world models},
  author={Bar, Amir and Zhou, Gaoyue and Tran, Danny and Darrell, Trevor and LeCun, Yann},
  journal={arXiv preprint arXiv:2412.03572},
  year={2024}
}
```

## Acknowledgments
We thank Noriaki Hirose for his help with the HuRoN dataset and for sharing his insights, and to Manan Tomar, David Fan, Sonia Joseph, Angjoo Kanazawa, Ethan Weber, Nicolas Ballas, and the anonymous reviewers for their helpful discussions and feedback.

## License
The code and model weights are licensed under Creative Commons Attribution-NonCommercial 4.0 International. See [`LICENSE.txt`](LICENSE.txt) for details.
