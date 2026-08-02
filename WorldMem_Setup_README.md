# Optional WorldMem setup

WorldMem is vendored only as a paper and implementation reference. The current
retrieval-based context-replacement experiment does not import it, reproduce
Minecraft, or require the WorldMem environment. Use the normal NWM profile:

```bash
bash setup_nwm_env.sh --profile nwm --backend cpu
source .venv-wsl/bin/activate
```

Only install the additional profile when explicitly assigned to reproduce an
upstream WorldMem/Minecraft experiment:

```bash
bash setup_nwm_env.sh --profile all --backend cu124
python worldmem_setup_and_test.py doctor
```

The Python helper auto-detects the repository root and defaults to read-only
diagnostics. It does not assume a maintainer-specific directory layout.
