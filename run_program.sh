./motion.exe
source ~/anaconda3/etc/profile.d/conda.sh
conda activate basics
python main.py --mode space --latent_dim 2 --num_epochs 1000 --num_cluster 2