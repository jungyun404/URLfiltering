import pandas as pd
import numpy as np
import requests
import re
from urllib.parse import urlparse
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Model
#from sklearn.metrics import confusion_matrix


# URL 특징 추출 함수
def url_to_embedding(url, max_length=100):
    # URL을 문자열로 인코딩하여 문자 단위로 분리
    url_str = str(url.encode('utf-8'))
    url_str = url_str.replace("\\n","").replace("b'","").replace("'","")
    url_chars = list(url_str)
    url_chars = [char.lower() for char in url_chars]
    # 각 문자에 대한 고유한 정수 인덱스 매핑
    char_index = {char: index+1 for index, char in enumerate(set(url_chars))}
    # URL 문자열을 고유한 정수 인덱스 시퀀스로 변환
    url_sequence = [char_index[char] for char in url_chars]
    # 시퀀스 패딩
    padded_sequence = pad_sequences([url_sequence], maxlen=max_length, padding='post', truncating='post')
    # URL 시퀀스 임베딩
    embedding_matrix = np.zeros((len(char_index)+1, 8))
    for i in range(1, len(char_index)+1):
        embedding_matrix[i,:] = np.random.normal(scale=0.6, size=(8,))
    url_embedding = np.zeros((1, max_length, 8))
    for i, char_index in enumerate(padded_sequence[0]):
        url_embedding[0, i, :] = embedding_matrix[char_index, :]
    # 임베딩된 URL을 반환
    return url_embedding

# 정상 URL과 피싱 URL 데이터셋 로드
legitimate_urls = pd.read_csv("legitimate_urls.csv")
phishing_urls = pd.read_csv("phishing_urls.csv")

# URL 특징 추출 및 데이터셋 생성
legitimate_urls["embedding"] = legitimate_urls["url"].apply(url_to_embedding)
phishing_urls["embedding"] = phishing_urls["url"].apply(url_to_embedding)

# 학습 데이터셋과 테스트 데이터셋 생성
train_legitimate, test_legitimate = np.split(legitimate_urls.sample(frac=1), [int(0.7 * len(legitimate_urls))])
train_phishing, test_phishing = np.split(phishing_urls.sample(frac=1), [int(0.7 * len(phishing_urls))])

# 학습 데이터셋과 테스트 데이터셋 병합
train_data = pd.concat([train_legitimate, train_phishing])
test_data = pd

# URL 이진 분류 모델 생성
input_layer = Input(shape=(train_data["embedding"].iloc[0].shape))
conv_layer = Conv1D(filters=128, kernel_size=3, activation='relu')(input_layer)
pooling_layer = GlobalMaxPooling1D()(conv_layer)
dense_layer = Dense(128, activation='relu')(pooling_layer)
dropout_layer = Dropout(0.5)(dense_layer)
output_layer = Dense(1, activation='sigmoid')(dropout_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습 데이터셋과 테스트 데이터셋 분리
train_X = np.array(train_data["embedding"].tolist())
train_y = np.array(train_data["label"].tolist())
test_X = np.array(test_data["embedding"].tolist())
test_y = np.array(test_data["label"].tolist())

# 모델 학습
history = model.fit(train_X, train_y, epochs=10, batch_size=64, validation_data=(test_X, test_y))

# 모델 평가
pred_y = model.predict(test_X)
pred_y = np.where(pred_y > 0.5, 1, 0)
conf_matrix = confusion_matrix(test_y, pred_y)
print("Confusion Matrix: ")
print(conf_matrix)
