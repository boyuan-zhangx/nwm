# 本地 WSL 环境

## 已验证的负责人机器

2026-08-02 实机验证：

- Ubuntu 22.04 / Python 3.10；
- PyTorch `2.4.1+cu124`；
- NVIDIA GeForce RTX 4060 Laptop GPU，CUDA runtime 可用；
- `doctor` 17/17、完整测试 11/11、实际 CUDA 矩阵运算通过。

Windows `.venv` 只用于 CPU 单测。WSL 与 Windows 不得共用同一个
virtualenv。

## 先决定磁盘布局

WSL distribution 默认把 `ext4.vhdx` 放在 C 盘。即使仓库位于
`/mnt/d`，Linux 的 `/home`、`/tmp` 和 pip cache 仍会占用 VHDX 所在
磁盘。C 盘耗尽会表现为安装无输出、`getpwnam failed` 或
`CreateInstance/E_FAIL`。

负责人机器采用以下布局：

- 仓库：`D:\Navware_workspace\nwm`；
- Ubuntu VHDX：`D:\WSL\Ubuntu-22.04`；
- WSL venv：`/home/zhang/.venvs/navware-nwm`；
- 仓库 `.venv-wsl`：指向上述 venv 的本地符号链接。

对已经安装且处于停止状态的发行版，可在 PowerShell 中迁移：

```powershell
wsl --shutdown
wsl --manage Ubuntu-22.04 --move D:\WSL\Ubuntu-22.04
```

执行前确认目标目录不存在、D 盘空间充足，并保留重要数据备份。

不要把大型 Linux venv 直接安装到 `/mnt/c` 或 `/mnt/d`。把 venv 放在
Linux 文件系统中，依赖安装和 import 都会更快。

## 首次创建 CUDA 环境

```bash
cd /mnt/d/Navware_workspace/nwm

sudo apt update
sudo apt install -y python3.10-venv ffmpeg

mkdir -p ~/.venvs
bash setup_nwm_env.sh \
  --profile nwm \
  --backend cu124 \
  --venv ~/.venvs/navware-nwm

# 仅在仓库中还不存在 .venv-wsl 时执行。
ln -s ~/.venvs/navware-nwm .venv-wsl

source .venv-wsl/bin/activate
python scripts/navware.py doctor --profile nwm
python scripts/navware.py smoke
```

安装成功后可以删除下载缓存，但不能删除 venv：

```bash
python -m pip cache purge
```

只有没有可见 NVIDIA GPU 的 CI/开发机才使用 `--backend cpu`。有 GPU 时
先运行 `nvidia-smi`，再选择仓库支持的 PyTorch wheel backend；不要根据
系统 CUDA toolkit 版本猜 wheel。

## Git ownership

仓库由 Windows Git clone 时，WSL Git 可能报告 `detected dubious
ownership`。由当前 WSL 用户执行一次：

```bash
git config --global --add safe.directory /mnt/d/Navware_workspace/nwm
```

## 日常使用

```bash
cd /mnt/d/Navware_workspace/nwm
source .venv-wsl/bin/activate

python scripts/navware.py doctor --profile nwm
python scripts/navware.py smoke
```

Windows 和 WSL 的激活入口分别是：

- Windows：`.venv/Scripts/python.exe`；
- WSL：`.venv-wsl/bin/python`。

## 配置路径，不改源码

```bash
cp config/paths.example.yaml config/paths.local.yaml
export NAVWARE_DATA_ROOT=/path/to/data
export NAVWARE_RESULTS_ROOT=/path/to/results
python scripts/navware.py doctor \
  --profile nwm \
  --config config/nwm_cdit_xl.yaml \
  --paths-config config/paths.local.yaml
```

`config/paths.local.yaml` 被 Git ignore。每台机器只有这一份本地路径文件，
实验 YAML 可以共享。
