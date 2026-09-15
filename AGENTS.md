## Environment

- This project has been setup to use `uv`.
- The machine you are working on has NVIDIA RTX 3060 available.
- Remote cluster has NVIDIA A100. The repo structure will be rsync-ed to stay identical as this local machine. Jobs are submitted with SLURM.

## Data folder

- Retinal OCT image datasets are in `datasets/` folder. They are git-ignored and require following symlinks. 
- Refer files in datasets by the symlink path, not the resolved location, to remain encapsulated when checked-out.
- There are two folders for storing results, `data/` for most cases, and `artifacts/` for intermediate results, which are git-ignored.

## Scratchpad folder

To perform experiment and/or store results, put inside the `.scratchpad` folder.
To organize files by each session, create or reuse existing subfolder with the name convention `[HEAD commit short SHA]_[slug]`. Do not write inside the scratchpad folder directly.

This folder should be use for storing ephemeral files only. For results that are final, direct outputs of the main pipeline, it should be stored separately using data folders above.

## Related research articles

- Index is provided in `docs/articles/index.md`. It is related to self-supervised retinal OCT classification.