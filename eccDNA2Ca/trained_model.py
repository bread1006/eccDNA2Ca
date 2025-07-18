from utils import eccDNA2Ca
import tensorflow as tf
import numpy as np
import  random

seed= # random seed
random.seed(int(seed))
np.random.seed(int(seed))
tf.random.set_seed(int(seed))
model = eccDNA2Ca(
    data_path="~/eccDNA2Ca/Selected_feature.xlsx",   ### xgboost feature matrix
    fasta_path="~/eccDNA2Ca/eccDNA.fasta",           ### eccDNA fasta file
    model_dir="models"
)
model.train()
