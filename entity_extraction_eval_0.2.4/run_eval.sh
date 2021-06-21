#!/bin/bash
source /nfs/home/vyasa/software/pkg/miniconda3/bin/activate
# conda init bash
conda activate scispacy_0.2.4

nvidia-smi
# python test/test.py
python -u entity_extraction_eval_0.2.4/entity_extraction_scispacy.py
python -u entity_extraction_eval_0.2.4/metrics_len.py
# sbatch -G 1 -o test/test.txt -w devbox4 run.sh
