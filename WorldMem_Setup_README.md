# Optional WorldMem setup

WorldMem is vendored as a reference implementation. LT-NWM does not import it at runtime, so the normal NWM profile is enough for LT-NWM development.

```bash
bash setup_nwm_env.sh --profile nwm --backend cpu
source .venv-wsl/bin/activate
```

Only when reproducing upstream WorldMem/Minecraft experiments:

```bash
bash setup_nwm_env.sh --profile all --backend cu124
python worldmem_setup_and_test.py doctor
```

The Python helper now auto-detects the repository root and defaults to read-only diagnostics. It no longer assumes the obsolete `Mine/nwm` and `Simon/nwm` directory layout.
