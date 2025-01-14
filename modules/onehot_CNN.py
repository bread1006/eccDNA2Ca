import re
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

seq=SeqIO.parse(open("D:/lina/papers/eccPred/Data_cancer/eccDNA.fasta"),
                'fasta')
sequences = [str(fasta.seq) for fasta in seq]
print(sequences[0])
# 创建 DataFrame，并保留索引
df = pd.DataFrame({'sequence': sequences})
df['index'] = df.index  # 保存原始索引

#one-hot coding
label_encoder = LabelEncoder()
label_encoder.fit_transform(np.array(['A','C','G','T']))
def string_to_array(my_string):
    my_string = re.sub('[^ACGT]', 'N', my_string)
    my_array = np.array(list(my_string))
    return my_array

def one_hot (array):
    integer_encoded = label_encoder.transform(array)
    onehot_encoder = OneHotEncoder(sparse_output=False, dtype=int)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

one_hot_matrices=[one_hot(string_to_array(seq)) for seq in sequences]
print(one_hot_matrices[0].shape)
#create a CNN model
import tensorflow as tf
import numpy as np
import random
import pandas as pd
from tensorflow.keras.layers import Masking,Reshape,Input,Conv1D, MaxPooling1D, Flatten, Dense, concatenate, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import (confusion_matrix,classification_report,
                             roc_auc_score,roc_curve,f1_score,recall_score,precision_score,
                             accuracy_score,precision_recall_curve,auc)
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# padding(value=0)
# max_length = max(len(seq) for seq in one_hot_matrices) #之前是24078，因为已知序列最长为24078，统一为25kb
max_length=25000
padded_matrices = pad_sequences(one_hot_matrices, maxlen=max_length, padding='post', dtype='float32',value=0)
label = np.array(pd.read_excel("D:/lina/papers/eccPred/Data_cancer/Data_cancer.xlsx",sheet_name="annotation").iloc[:, -1])

seed=333
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
X_train,X_test,y_train,y_test, train_index, test_index=train_test_split(padded_matrices,label, df['index'],
                                                                        test_size=0.2,
                                                                        random_state=seed)
unique, counts = np.unique(y_test, return_counts=True)
print(unique,counts)
print(f"测试集形状: {X_test.shape}")
print(f"训练集形状:{X_train.shape}")
print(np.unique(y_train,return_counts=True))
print("Train indices:", list(train_index))
print("Test indices:", list(test_index))

# CNN
def get_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        # Conv1D(16, kernel_size=3, activation='relu', kernel_initializer='he_normal'),
        # MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=3, activation='relu',kernel_initializer='he_normal'),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation='relu',kernel_initializer='he_normal'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu',kernel_initializer='he_normal'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001),loss="binary_crossentropy",metrics=['accuracy'])
    return model

epochs = 50
batch_size = 32
npatience = 20

# # cross-validation
# kf = KFold(n_splits=5, shuffle=True, random_state=123)
# fold = 1
# all_train_losses = []
# all_val_losses = []
# accs= []
# f1_scores = []
# precisions = []
# recalls = []
# aucs = []
# auprs= []
# for train_index, val_index in kf.split(X_train):
#     print(f"Train fold {fold}...")
#
#     X_tr, X_val = X_train[train_index], X_train[val_index]
#     y_tr, y_val = y_train[train_index], y_train[val_index]
#
#     model = get_cnn_model(input_shape=X_train.shape[1:])
#
#     history = model.fit(X_tr, y_tr,
#                         epochs=epochs,
#                         batch_size=batch_size,
#                         validation_data=(X_val, y_val),
#                         callbacks=[EarlyStopping(patience=npatience,
#                                                  monitor='val_loss',
#                                                  mode='min')],
#                         verbose=0)
#     all_train_losses.append(history.history['loss'])
#     all_val_losses.append(history.history['val_loss'])
#
#     loss, accuracy = model.evaluate(X_val,  y_val)
#     accs.append(round(accuracy,4))
#
#     # 在验证集上进行评估
#     val_predictions = model.predict(X_val)
#     val_pred_labels = (val_predictions > 0.5).astype(int)
#     precision_vals, recall_vals, _ = precision_recall_curve(y_val, val_predictions)
#
#     # 计算评估指标
#     auc_score = round(roc_auc_score(y_val, val_predictions),4)
#     f1 = round(f1_score(y_val, val_pred_labels),4)
#     precision = round(precision_score(y_val, val_pred_labels),4)
#     recall = round(recall_score(y_val, val_pred_labels),4)
#     aupr = round(auc(recall_vals, precision_vals),4)
#
#     # 存储每折的评估结果
#     aucs.append(auc_score)
#     f1_scores.append(f1)
#     precisions.append(precision)
#     recalls.append(recall)
#     auprs.append(aupr)
#     print(f"Fold {fold} Validation Accuracy: {accuracy:.4f}")
#     fold += 1
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
# plt.xticks(np.arange(0,epochs,step=1))
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# 训练最终模型
seed=123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
final_model=get_cnn_model(X_train.shape[1:])
hist=final_model.fit(X_train,y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[EarlyStopping(patience=npatience, monitor='loss', mode='min')],
                verbose=0)
plt.plot(hist.history['loss'], label='loss')
plt.xticks(np.arange(0,epochs,step=1))
plt.legend()
plt.show()

#测试集预测
test_loss, test_accuracy = final_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

y_train_pred = final_model.predict(X_train)
y_train_pred_class = (y_train_pred > 0.5).astype(int)
train_pred_positive_indices = np.where(y_train_pred_class.flatten() == y_train)[0]
train_pred_positive_original_indices = train_index.iloc[train_pred_positive_indices].values
print("Indices of training samples predicted correctly:", train_pred_positive_original_indices)

y_pred = final_model.predict(X_test)
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

np.save('CNN_y_test.npy', y_test)
np.save('CNN_y_pred.npy', y_pred)