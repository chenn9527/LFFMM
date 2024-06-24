
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from models import *
from sklearn.metrics import classification_report, confusion_matrix

from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import metrics
import models

number = 10   #第几个预测音频
num_fold = 1
num_itera = 12
model_name = "LMF-CSR"
feature1 = 'mgl'
feature2 = 'spec'

audio_path = f'chick_sound/audio_predict/predict{number}/predict_audio'
metadata_path = f'chick_sound/audio_predict/predict{number}/chick_sound.csv'
metadata = pd.read_csv(metadata_path)
#加载特征
# 信号特征
X_signal = np.load(f"extracted_features/chick_feat/X-predict{number}_mgl.npy")
Y = np.load(f"extracted_features/chick_feat/Y-predict{number}_mgl.npy")
X_spec = np.load(f"extracted_features/chick_feat/X-predict{number}_spec.npy")

label_encoder = LabelEncoder()  #one-hot编码
Y_encoded = to_categorical(label_encoder.fit_transform(Y))  #这里要用fit_transform
#测试集用预测数据集
X_test_signal = X_signal
X_test_spec = X_spec
Y_test_encoded = Y_encoded


#训练集用大训练集
X_train = np.load(f"extracted_features/chick_feat/X-mgl.npy")
Y_train = np.load(f"extracted_features/chick_feat/Y-mgl.npy")
X_spec_train = np.load(f"extracted_features/chick_feat/X-spectrogram.npy")
Y_encoded_train = to_categorical(label_encoder.fit_transform(Y_train))
test_index, train_index = models.get_fold(num_fold)
# X_test_signal = np.take(X_signal, test_index, axis=0)  # 信号特征测试集
# X_test_spec = np.take(X_spec, test_index, axis=0)  # 频谱图测试集
X_train_signal = np.take(X_train, train_index, axis=0)  # 信号特征训练集
X_train_spec = np.take(X_spec_train, train_index, axis=0)  # 频谱图训练集
Y_train_encoded = np.take(Y_encoded_train, train_index, axis=0)

#加载mgcs模型
custom_objects = {'SpatialAttention': SpatialAttention}
model_signal = models.load_model(f'E:/save_model/{model_name}/{feature1}/fold{num_fold}_{num_itera}_model_signal.h5',custom_objects=custom_objects)
model_spec = models.load_model(f'E:/save_model/{model_name}/{feature1}/fold{num_fold}_{num_itera}_model_spec.h5',custom_objects=custom_objects)
model_signal.summary()
model_spec.summary()
#模型预测

# 1. 使用信号模型和图片模型进行预测
features_signal_train = model_signal.predict(X_train_signal)
features_image_train = model_spec.predict(X_train_spec)
features_signal_test = model_signal.predict(X_test_signal)
features_image_test = model_spec.predict(X_test_spec)

# 2. 将两个模型的特征连接成一个新的特征向量
combined_train_features = np.concatenate((features_signal_train, features_image_train), axis=1)
combined_test_features = np.concatenate((features_signal_test, features_image_test), axis=1)

# SVM分类器

Y_train = np.argmax(Y_train_encoded, axis=1)
svm_model = SVC(kernel='linear', C=40, gamma=0.0001)
svm_model.fit(combined_train_features, Y_train)
        # 在测试集上进行预测
y_pred = svm_model.predict(combined_test_features)      #获得预测值
        # svm计算准确率
train = svm_model.score(combined_train_features, Y_train)
test = svm_model.score(combined_test_features, np.argmax(Y_test_encoded, axis=1))
unique_labels, label_counts = np.unique(y_pred, return_counts=True)
for label, count in zip(unique_labels, label_counts):
    print(f"Class {label}: {count} samples")



y_trues = np.argmax(Y_test_encoded, axis=1)
incorrect_indices = np.where(y_pred != y_trues)[0]
# 打印分类错误的样本
test_index = models.get_predict_index(104)
for idx in incorrect_indices:
    print(f"样本索引: {idx},在csv文件中的位置:{test_index[idx]},名字为{metadata.iloc[test_index[idx],0]} 真实标签: {y_trues[idx]}, 预测标签: {y_pred[idx]}")

# CM = confusion_matrix(y_trues, y_pred)
labels =['cough', 'crow', 'feeder', 'flap', 'normal', 'peck', 'scream','snore','ventilator']
re = classification_report(y_trues, y_pred, labels=[0,1,2,3,4,5,6,7,8], target_names=labels,digits=5)
metrics.display_cm_predict(y_trues,y_pred,number)
print(re)
