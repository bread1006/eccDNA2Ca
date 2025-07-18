from utils import eccDNA2Ca
import tensorflow as tf
import numpy as np
import  random

seed= 505
random.seed(int(seed))
np.random.seed(int(seed))
tf.random.set_seed(int(seed))
model = eccDNA2Ca(
    data_path="D:/lina/files/PycharmProjects/pythonProject/eccDNA2Ca/Selected_feature.xlsx",
    fasta_path="D:/lina/files/PycharmProjects/pythonProject/eccDNA2Ca/eccDNA.fasta",
    model_dir="models"
)
model.train()