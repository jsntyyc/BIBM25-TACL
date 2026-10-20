# TACL medical text understanding research code

Code accompanying **TACL: Threshold-Adaptive Curriculum Learning Strategy for Enhancing Medical Text Understanding**, BIBM 2025 Regular paper. [Paper](https://arxiv.org/abs/2510.15269v2) · [中文指南](README.zh-CN.md) · [Upstream](https://github.com/jsntyyc/BIBM25-TACL)

This checkout reorganizes upstream commit `67306350a049daf87ef5ac384b367c87d5b349eb` while preserving its Git history and experiment algorithms. It is a maintenance refactor, not a reproduction of the paper's reported results. Known differences between the released scripts and the paper are documented in [implementation audit](docs/implementation_audit.md).

| Task | Entry and documentation | Included assets |
|---|---|---|
| ICD coding | `python -m tasks.icd.main --help`; [guide](tasks/icd/README.md) | Source, Hydra configs, historic experiments, split/subset Feather files |
| Readmission | `python -m tasks.readmission.run_readmission --help`; [guide](tasks/readmission/README.md) | ClinicalBERT and baseline scripts; no complete dataset or pretrained weights |
| TCM syndrome | `python -m tasks.tcm.run --help`; [guide](tasks/tcm/README.md) | ZY-BERT experiments, partial provided data, raw-format examples |
| Out-of-hospital mortality | [audit](docs/implementation_audit.md) | A complete paper-matched pipeline has not been established |

## Start here

Use Python 3.10 or later. Run the commands from this repository root. An editable installation (`python -m pip install -e . --no-deps`) makes the module entrypoints accessible from other working directories. Use separate task environments; the old ClinicalBERT and newer ICD stacks have different dependency requirements. Install the task's requirements only when preparing that environment. Root installation intentionally does not install GPU libraries.

```console
python -m tasks.icd.main --help
python -m tasks.tcm.run --help
python -m tasks.readmission.run_readmission --help
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

Help, path checks and repository checks do not train a model or fetch patient data. Actual training requires externally prepared datasets and model weights; see [data and models](docs/data_and_models.md). The bundled TCM examples are not ready-made `TEXT`/`LABEL` training files.

## Layout

- `tasks/`: task-local source, configurations, legacy experiments and lazy entrypoints.
- `data/`: provided TCM assets and ICD splits, plus locations for external datasets.
- `models/`, `runs/`: external weights and generated outputs, ignored by Git.
- `archives/`: unmodified upstream ZIP archives, including overlapping historic TCM code.
- `docs/upstream/`: original upstream documentation and scoped license.
- `docs/migration.csv`: every original file/archive member, destination, SHA-256 and generated-file omissions.
- `docs/research/`: literature, proposed follow-up method and planned experiments. Research hypotheses are not measured improvements.

The [Chinese research proposal (PDF)](docs/research/TACL-research-proposal.pdf) includes 22 core references, the mathematical formulation, and a staged experimental plan for a single GPU with approximately 48 GB of memory.

## Provenance and licensing

See [third-party notices](THIRD_PARTY_NOTICES.md). Preserve the distinct license scopes of copied source and data. This reorganization does not assign a new license to the entire upstream repository. Citation metadata is in [CITATION.cff](CITATION.cff).
