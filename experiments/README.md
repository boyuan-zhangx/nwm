# Experiment manifests

These YAML files freeze scientific comparisons independently of Python code.
They are specifications until a runner explicitly validates and consumes every
field. A manifest must never be presented as an executed experiment by itself.

Rules:

- the same seeds, checkpoint, sampler steps and data manifest apply to all groups;
- group selection is a config value, never a source-code edit;
- every run saves retrieval diagnostics and resolved configuration;
- missing required output invalidates the run.
