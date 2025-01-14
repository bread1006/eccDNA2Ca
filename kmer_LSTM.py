import numpy as np
from Bio import SeqIO
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split,KFold
from sklearn.feature_extraction import FeatureHasher
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense,Input,Bidirectional,GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import (roc_curve, roc_auc_score,
                             f1_score, recall_score,
                             precision_score,auc,
                             classification_report, confusion_matrix,accuracy_score,precision_recall_curve)

seq = SeqIO.parse(open("D:/lina/papers/eccPred/Data_cancer/eccDNA.fasta"),"fasta")
sequences = [str(fasta.seq) for fasta in seq]
# 创建 DataFrame，并保留索引
df = pd.DataFrame({'sequence': sequences})
df['index'] = df.index  # 保存原始索引
def generate_kmers(sequence,k):
    kmers=[sequence[i:i+k] for i in range (len(sequence)-k+1)]
    return kmers

k=4
kmer_list = [generate_kmers(seq,k) for seq in sequences]
vectorizer = FeatureHasher(n_features=4**k, input_type="string")
hashed_vectors = vectorizer.transform(kmer_list).toarray()

label = np.array(pd.read_excel("D:/lina/papers/eccPred/Data_cancer/Data_cancer.xlsx",sheet_name="annotation").iloc[:, -1])
hashed_vectors=np.expand_dims(hashed_vectors,axis=-1)
print(hashed_vectors.shape)

seed=333
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(hashed_vectors, label,df['index'],
                                                    test_size=0.2,
                                                    random_state=seed)
unique, counts = np.unique(y_test, return_counts=True)
print(unique,counts)

def get_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64,return_sequences=True,kernel_regularizer=l2(l2_param)),
        LSTM(64,kernel_regularizer=l2(l2_param)),
        Dense(32, activation="tanh",kernel_regularizer=l2(l2_param)),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape=hashed_vectors.shape[1:]
epochs=60
batch_size=32
npatience=20
lr=0.0001
l2_param=0.0001
early_stopping = EarlyStopping(monitor='val_loss', patience=npatience, mode="min")

# # cross validation
# kf=KFold(n_splits=5,shuffle=True,random_state=42)
# fold=1
#
# all_train_losses = []
# all_val_losses = []
# accs= []
# f1_scores = []
# precisions = []
# recalls = []
# aucs = []
# auprs= []
# for train_index,val_index in kf.split(X_train):
#     print(f"train fold {fold}")
#     X_tr,X_val=X_train[train_index],X_train[val_index]
#     y_tr,y_val=y_train[train_index],y_train[val_index]
#
#     model=get_lstm_model(input_shape=input_shape)
#     history=model.fit(X_tr,y_tr,
#                       epochs=epochs,
#                       batch_size=batch_size,
#                       validation_data=(X_val,y_val),
#                       callbacks=[early_stopping],
#                       verbose=0
#                       )
#     all_train_losses.append(history.history['loss'])
#     all_val_losses.append(history.history['val_loss'])
#
#     loss,accuracy=model.evaluate(X_val,y_val)
#     accs.append(round(accuracy,4))
#     # 在验证集上进行评估
#     val_predictions = model.predict(X_val)
#     val_pred_labels = (val_predictions > 0.5).astype(int)
#     precision_vals, recall_vals, _ = precision_recall_curve(y_val, val_predictions)
#     # 计算评估指标
#     auc_score = round(roc_auc_score(y_val, val_predictions),4)
#     f1 = round(f1_score(y_val, val_pred_labels),4)
#     precision = round(precision_score(y_val, val_pred_labels),4)
#     recall = round(recall_score(y_val, val_pred_labels),4)
#     aupr = round(auc(recall_vals, precision_vals),4)
#     # 存储每折的评估结果
#     f1_scores.append(f1)
#     precisions.append(precision)
#     recalls.append(recall)
#     aucs.append(auc_score)
#     auprs.append(aupr)
#
#     print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
#     fold+=1
#
# print(f"val_auc: {aucs}")
# print(f"val_acc: {accs}")
# print(f"val_f1: {f1_scores}")
# print(f"val_precision: {precisions}")
# print(f"val_recall: {recalls}")
# print(f"val_aupr: {auprs}")
# # 绘制训练和验证损失曲线
# average_train_losses = np.nanmean([np.pad(losses, (0, max(map(len, all_train_losses)) - len(losses)), 'constant',constant_values=np.nan) for losses in all_train_losses], axis=0)
# average_val_losses = np.nanmean([np.pad(losses, (0, max(map(len, all_val_losses)) - len(losses)), 'constant',constant_values=np.nan) for losses in all_val_losses], axis=0)
# plt.plot(average_train_losses, label='Training Loss')
# plt.plot(average_val_losses, label='Validation Loss')
# plt.xticks(np.arange(epochs,step=1))
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# final model
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
lstm_model = get_lstm_model(input_shape)
hist = lstm_model.fit(X_train, y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 verbose=0,
                 shuffle=False,
                 callbacks=EarlyStopping(monitor='loss', patience=npatience, mode="min"))

plt.plot(hist.history['loss'], label='loss')
plt.xticks(np.arange(epochs,step=1))
plt.legend()
plt.show()

# 评估模型
test_loss, test_accuracy = lstm_model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_accuracy:.4f}")

y_train_pred = lstm_model.predict(X_train)
y_train_pred_class = (y_train_pred > 0.5).astype(int)
train_pred_positive_indices = np.where(y_train_pred_class.flatten() == y_train)[0]
train_pred_positive_original_indices = train_index.iloc[train_pred_positive_indices].values
print("Indices of training samples predicted correctly:", train_pred_positive_original_indices)

y_pred = lstm_model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)
test_pred_positive_indices = np.where(y_pred_class.flatten() == y_test)[0]
test_pred_positive_original_indices = test_index.iloc[test_pred_positive_indices].values
print("Indices of test samples predicted correctly:", test_pred_positive_original_indices)


auc_score = roc_auc_score(y_test, y_pred)
Precision, Recall, _ = precision_recall_curve(y_test, y_pred)
aupr = auc(Recall, Precision)
accuracy=accuracy_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
print(f"AUC:{auc_score:.4f}")
print(f"accuracy_score:{accuracy:.4f}")
print(f"f1_score:{f1:.4f}")
print(f"precision_score:{precision:.4f}")
print(f"recall_score:{recall:.4f}")
print(f"aupr:{aupr:.4f}")
print(classification_report(y_test, y_pred_class))
print(confusion_matrix(y_test, y_pred_class))

np.save('LSTM_y_test.npy', y_test)
np.save('LSTM_y_pred.npy', y_pred)