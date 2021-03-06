#!/bin/bash
source /nfs/home/vyasa/software/pkg/miniconda3/bin/activate
# conda init bash
conda activate scispacy_0.4

nvidia-smi
# python test/test.py
#python -u entity_extraction_comp/entity_cui_extraction_scispacy.py
#python -u entity_extraction_comp/metrics_len.py
#python -u misc/clustering/tsne_org_comb.py
python -u misc/clustering/lsh_comb.py
#python -u misc/clustering/lsh.py
#python -u misc/tsne_plot_org.py
#python -u misc/clustering/lsh.py
#python -u misc/clustering/dbscan.py
# sbatch -G 1 -o test/test.txt -w devbox4 run.sh
