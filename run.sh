#!/bin/bash
source /nfs/home/vyasa/software/pkg/miniconda3/bin/activate
# conda init bash
conda activate scispacy

nvidia-smi
# python test/test.py
python -u entity_extraction/entity_extraction_scispacy.py
python -u metrics/metrics.py
# sbatch -G 1 -o test/test.txt -w devbox4 run.sh
