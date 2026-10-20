# Third-party provenance and license scopes

The complete upstream Git history is retained. The baseline is `67306350a049daf87ef5ac384b367c87d5b349eb` from `jsntyyc/BIBM25-TACL`. This maintenance work does not grant rights beyond the licenses supplied upstream.

- The original ZY-BERT MIT license is retained at `docs/upstream/tcm/LICENSE`; its original README is beside it. That README describes TCM-SD data as CC BY-NC-SA 4.0. Treat data terms separately from source-code terms.
- ClinicalBERT-derived source files contain Apache License 2.0 notices. Those notices remain in the implementation files. The upstream ClinicalBERT README is preserved in `docs/upstream/readmission/README.md`.
- The upstream repository has no root license covering the entire combined project. No blanket MIT/Apache license has been added for ICD, contributed experiment variants, or the aggregate repository.
- MIMIC data, pretrained model weights, terminology resources and external libraries retain their own access conditions and licenses. The repository does not redistribute newly downloaded patient records or weights.
- `archives/` retains original ZIP bytes for provenance. Cached/temporary members excluded from the active source tree remain available in those archives and/or Git history.
