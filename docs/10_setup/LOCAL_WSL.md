# Local WSL Setup

This environment supports Phase A inference and evaluation. Phase A does not
require WorldMem dependencies, `HybridCDiT` training, or a full NWM retrain.

## Verified reference environment

The maintainer verified the following combination on 2026-08-02:

- Ubuntu 22.04 and Python 3.10;
- PyTorch `2.4.1+cu124`;
- an NVIDIA RTX 4060 visible from WSL;
- doctor 17/17, the repository tests, and a real CUDA tensor operation.

This is a reference, not a required personal path layout. Windows and WSL must
not share a virtual environment. A Windows venv contains Windows executables;
a WSL venv contains Linux executables.

## Storage layout

WSL stores its Linux filesystem in an `ext4.vhdx`. By default that virtual disk
is usually on the Windows system drive. A repository on a mounted Windows drive
does not change where Linux `/home`, `/tmp`, or the pip cache are stored. When
the system drive fills, WSL can fail with misleading errors such as
`getpwnam failed` or `CreateInstance/E_FAIL`.

Recommended layout:

1. Move the WSL distribution to a drive with sufficient free space if needed.
2. Clone the repository inside the Linux filesystem, for example under
   `~/src/nwm`, for the best small-file performance.
3. Keep the venv inside the Linux filesystem.
4. Keep large datasets and outputs on an appropriate data drive or cluster
   filesystem and expose them through `config/paths.local.yaml`.

Recent WSL versions support distribution migration from PowerShell:

```powershell
wsl --shutdown
wsl --manage Ubuntu-22.04 --move <large-drive>:\WSL\Ubuntu-22.04
```

Replace `<large-drive>` before running the command. Confirm that the target does
not already exist, that the destination has enough space, and that important
Linux data is backed up.

## Install the CUDA environment

From an Ubuntu shell:

```bash
sudo apt update
sudo apt install -y python3.10-venv ffmpeg git

mkdir -p ~/src
cd ~/src
git clone https://github.com/boyuan-zhangx/nwm.git
cd nwm

bash setup_nwm_env.sh --profile nwm --backend cu124
source .venv-wsl/bin/activate

python scripts/navware.py doctor --profile nwm
python scripts/navware.py smoke
python -m pytest -q
```

The NWM profile pins `accelerate`, `diffusers`, and `transformers` to the
versions validated with PyTorch 2.4.1. If doctor reports a version failure,
rerun `bash setup_nwm_env.sh --profile nwm --backend cu124`; do not silence the
check or upgrade only one Hugging Face package.

Use `--backend cpu` only on a machine without a visible NVIDIA GPU. On a GPU
machine, run `nvidia-smi` first and select one of the backends supported by the
setup script. The locally installed CUDA toolkit version does not select the
PyTorch wheel; the driver and wheel compatibility do.

The RTX 4060 is the primary CDiT/S development machine for environment checks,
manifest construction, context-policy tests, and the small oracle gate. Begin
with batch size one and mixed precision, then measure actual memory use. CDiT/B
is optional locally. Reserve CDiT/XL confirmation for the cluster after the
CDiT/S oracle and pose-aligned gates pass.

After a successful installation, the wheel download cache may be removed:

```bash
python -m pip cache purge
```

Do not delete the venv itself.

## Repository on a mounted Windows drive

This layout is supported but slower for dependency installation. Put the venv
under Linux home and tell the setup script where to create it:

```bash
cd /path/to/mounted/repository
export NAVWARE_VENV="$HOME/.venvs/navware-nwm"
mkdir -p "$(dirname "$NAVWARE_VENV")"

bash setup_nwm_env.sh \
  --profile nwm \
  --backend cu124 \
  --venv "$NAVWARE_VENV"

source "$NAVWARE_VENV/bin/activate"
export NAVWARE_PYTHON="$NAVWARE_VENV/bin/python"
```

The train and inference wrappers honor the active `VIRTUAL_ENV`, active Conda
environment, and `NAVWARE_PYTHON`; a repository-local venv is not required.

If Git reports `detected dubious ownership` for a Windows-owned repository,
run this once from the repository root:

```bash
git config --global --add safe.directory "$(pwd)"
```

## Configure paths without editing source

```bash
cp config/paths.example.yaml config/paths.local.yaml
```

Edit the copied file, then validate it:

```bash
python scripts/navware.py doctor \
  --profile nwm \
  --config config/nwm_cdit_xl.yaml \
  --paths-config config/paths.local.yaml
```

`config/paths.local.yaml` is ignored by Git. Each machine has one local overlay;
experiment YAML files remain portable and reviewable.
