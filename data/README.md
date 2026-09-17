# Dataset provenance

Raw data is downloaded locally; it is NOT republished in this public repository.
Use the original sources and respect their dataset-specific terms. A repository's
code license does not automatically license all photographs or footage.

| Dataset | Source | Purpose | Limits |
|---|---|---|---|
| ShanghaiTech Part B | https://github.com/desenzhou/ShanghaiTechDataset | Train and evaluate the MobileNetV2 density model | Still images; not temporal/risk labels. Official 400 training / 316 test images; split training further for validation. |
| UMN crowd activity demonstration | https://mha.cs.umn.edu/proj_events.shtml | End-to-end functional video test | The university's AVI is a demonstration, not a verified raw, labelled benchmark. Do not claim anomaly accuracy from it. |
| COCO | https://cocodataset.org/ | Provenance of pretrained YOLOv8 detector | No COCO download or detector fine-tuning claimed. |

Run `python scripts/download_data.py`. Download receipts include byte counts and
SHA-256 checksums. If a source fails, the program exits unsuccessfully and reports
the error; it never substitutes synthetic data without disclosure.

Forecast training needs real chronological count sequences with independent
recording IDs. `training/train_forecast.py` accepts such data; shuffled
ShanghaiTech photographs must never be passed off as video sequences.

UCF-QNRF (https://www.crcv.ucf.edu/data/ucf-qnrf/), CrowdHuman
(https://www.crowdhuman.org/) and PETS2009 are future optional benchmarks, not
datasets this version claims to have trained on.
