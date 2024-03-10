#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
import numpy as np
import time
from gensim import corpora
import jieba
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import shap
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Embedding, Dense, LSTM, GRU, Bidirectional, Reshape, ReLU, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Concatenate, Dropout, BatchNormalization, Layer, Conv1D
from transformers import BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K
import random
import transformers
import scipy as sp
import matplotlib.font_manager as fm
import seaborn as sns
from tensorflow.keras.optimizers import Adam
plt.rcParams['font.family'] = 'SimSun'


# In[ ]:


准备数据
# 读取Excel表格数据，并将其转换为DataFrame
data = pd.read_excel('/...')
frame_x = pd.DataFrame(data,columns=['日期','性别','年龄','主诉','诊断'])
#日期处理
np.random.seed(42)
# 生成日期的数据格式
range_of_dates = pd.date_range(start="2017.01.01",end="2022.12.31")
X = pd.DataFrame(index=range_of_dates)
# 创建日期数据的序列
X["day_nr"] = range(len(X))
X["day_of_year"] = X.index.day_of_year
# 生成目标成分
signal_1 = 3 + 4 * np.sin(X["day_nr"] / 365 * 2 * np.pi)
signal_2 = 3 * np.sin(X["day_nr"] / 365 * 4 * np.pi + 365 / 2)
# 合并获取目标序列
y = signal_1 + signal_2
results_df = y.to_frame()
results_df.columns = ["actuals"]
frame_x = frame_x.iloc[:,[0,1,2,3]]
frame_x['日期'] = pd.to_datetime(frame_x['日期'])
frame_x.set_index('日期', inplace=True)
frame_x.insert(0, '日期', results_df.loc[frame_x.index, 'actuals'].values)

#年龄处理
frame_x['年龄'] = frame_x['年龄'].str.replace(r'[\u4e00-\u9fa5]+', '', regex=True)
print(frame_x.head(5))
data_file = np.array(frame_x)
#性别处理
data_file[0:][data_file[0:] == '男'] = 1
data_file[0:][data_file[0:] == '女'] = 0

data_encoder = np.array(data_final[['日期','性别','年龄']])
print(data_encoder.shape)#(362347, 3)


# In[ ]:


#现在拿我们的语料库来训练glove，得到自己的词向量表
#编写GloVe模型
def randmatrix(m, n):
    """Creates an m x n matrix of random values drawn using
    the Xavier Glorot method."""
    val = np.sqrt(6.0 / (m + n))
    return np.random.uniform(-val, val, size=(m, n))


def log_of_array_ignoring_zeros(M):
    log_M = M.copy()
    mask = log_M > 0
    log_M[mask] = np.log(log_M[mask])
    return log_M


def noise(n, scale=0.01):
    return np.random.normal(0, scale, size=n)


class AdaGradOptimizer:
    def __init__(self, learning_rate, initial_accumulator_value=0.1,momentum=None):
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self._momentum = momentum

    def get_step(self, grad):
        if self._momentum is None:
            self._momentum = self.initial_accumulator_value * np.ones_like(grad)
        self._momentum += grad ** 2
        return self.learning_rate * grad / np.sqrt(self._momentum)


class GloVe(object):
    def __init__(self, n, max_iter, learning_rate):
        self.n = n
        self.max_iter = max_iter
        self.xmax = 100
        self.alpha = 0.75
        self.mittens = 0
        self.learning_rate = learning_rate
        self.tol = 1e-4
        self.display_progress = 100
        self.model = None
        self.n_words = None
        self.log_dir = None
        self.log_subdir = None
        self.errors = list()
        self.test_mode = False

    def _initialize(self, coincidence):
        self.n_words = coincidence.shape[0]
        bounded = np.minimum(coincidence, self.xmax)
        weights = (bounded / float(self.xmax)) ** self.alpha
        log_coincidence = log_of_array_ignoring_zeros(coincidence)
        return weights, log_coincidence

    def fit(self, X, vocab=None, initial_embedding_dict=None):
        weights, log_coincidence = self._initialize(X)
        self._initialize_w_c_b(self.n_words, vocab, initial_embedding_dict)
        m_loop = tqdm(range(self.max_iter))
        for iteration in m_loop:
            pred = self._make_prediction()
            gradients, error = self._get_gradients_and_error(
                pred, log_coincidence, weights)
            self.errors.append(error)
            self._apply_updates(gradients)
            m_loop.set_description("Iteration {}:error {:4.4f}".format(iteration + 1, error))
        return self.W + self.C

    def _check_shapes(self, gradients):
        assert gradients['W'].shape == self.W.shape
        assert gradients['C'].shape == self.C.shape
        assert gradients['bw'].shape == self.bw.shape
        assert gradients['bc'].shape == self.bc.shape

    def _initialize_w_c_b(self, n_words, vocab, initial_embedding_dict):
        self.W = randmatrix(n_words, self.n)  # Word weights.
        self.C = randmatrix(n_words, self.n)  # Context weights.
        if initial_embedding_dict:
            assert self.n == len(next(iter(initial_embedding_dict.values())))

            self.original_embedding = np.zeros((len(vocab), self.n))
            self.has_embedding = np.zeros(len(vocab), dtype=bool)

            for i, w in enumerate(vocab):
                if w in initial_embedding_dict:
                    self.has_embedding[i] = 1
                    embedding = np.array(initial_embedding_dict[w])
                    self.original_embedding[i] = embedding
                    # Divide the original embedding into W and C,
                    # plus some noise to break the symmetry that would
                    # otherwise cause both gradient updates to be
                    # identical.
                    self.W[i] = 0.5 * embedding + noise(self.n)
                    self.C[i] = 0.5 * embedding + noise(self.n)
            # This is for testing. It differs from
            # `self.original_embedding` only in that it includes the
            # random noise we added above to break the symmetry.
            self.G_start = self.W + self.C

        self.bw = randmatrix(n_words, 1)
        self.bc = randmatrix(n_words, 1)
        self.ones = np.ones((n_words, 1))

    def _make_prediction(self):
        # Here we make use of numpy's broadcasting rules
        pred = np.dot(self.W, self.C.T) + self.bw + self.bc.T
        return pred

    def _get_gradients_and_error(self,
                                 predictions,
                                 log_coincidence,
                                 weights):
        # First we compute the GloVe gradients
        diffs = predictions - log_coincidence
        weighted_diffs = np.multiply(weights, diffs)
        wgrad = weighted_diffs.dot(self.C)
        cgrad = weighted_diffs.T.dot(self.W)
        bwgrad = weighted_diffs.sum(axis=1).reshape(-1, 1)
        bcgrad = weighted_diffs.sum(axis=0).reshape(-1, 1)
        error = (0.5 * np.multiply(weights, diffs ** 2)).sum()

        # Then we add the Mittens term (only if mittens > 0)
        if self.mittens > 0:
            curr_embedding = self.W + self.C
            distance = curr_embedding[self.has_embedding, :] -                        self.original_embedding[self.has_embedding, :]
            wgrad[self.has_embedding, :] += 2 * self.mittens * distance
            cgrad[self.has_embedding, :] += 2 * self.mittens * distance
            error += self.mittens * (
                    np.linalg.norm(distance, ord=2, axis=1) ** 2).sum()
        return {'W': wgrad, 'C': cgrad, 'bw': bwgrad, 'bc': bcgrad}, error

    def _apply_updates(self, gradients):
      
        if not hasattr(self, 'optimizers'):
            self.optimizers =                 {obj: AdaGradOptimizer(self.learning_rate)
                 for obj in ['W', 'C', 'bw', 'bc']}
        self.W -= self.optimizers['W'].get_step(gradients['W'])
        self.C -= self.optimizers['C'].get_step(gradients['C'])
        self.bw -= self.optimizers['bw'].get_step(gradients['bw'])
        self.bc -= self.optimizers['bc'].get_step(gradients['bc'])
#对中文语料库进行分词和获得共现矩阵
def leftRight(c_pos, max_len, window):
    return c_pos - window if c_pos - window > 0 else 0,            c_pos + window + 1 if c_pos + window + 1 < max_len else max_len


def getCoMatriex(texts, token_id, window=2):
    n_matrix = len(token_id)
    word_matrix = np.zeros(shape=[n_matrix, n_matrix])

    for i in range(len(texts)):
        k = len(texts[i])
        for j in range(k):
            left, right = leftRight(j, k, window)
            c_word = texts[i][j]
            c_pos = token_id[c_word]
            for m in range(left, right):
                # 计算共现矩阵
                t_word = texts[i][m]
                t_pos = token_id[t_word]
                if m != j and t_word != c_word:
                    word_matrix[c_pos][t_pos] += 1
    return word_matrix


def getCorpora(texts):
    dct = corpora.Dictionary(texts)
    token2idDict = dct.token2id
    return dct, token2idDict
import re
#模型训练
if __name__ == "__main__":
    # 生成词汇相关矩阵
    data = data_final.iloc[:, 3]

    texts = []
    for text in data:
             # 分词并去除标点符号
        words = jieba.lcut(text)
        words_without_punctuation = []
        for word in words:
            # 使用正则表达式去除标点符号
            word_without_punctuation = re.sub(r'[^\w\s]', '', word)
            # 排除空字符
            if word_without_punctuation.strip() != "":
                words_without_punctuation.append(word_without_punctuation)
        texts.append(words_without_punctuation)
    n_dims = 20
    print(texts)

     # 获得语料字典
    dct, token_id = getCorpora(texts) 

    stratTime = time.time()
    # 计算共现矩阵
    wordComatrix = getCoMatriex(texts, token_id, window=2)
    print("total time cost:", time.time() - stratTime)

    #print(word_matrix)
    # 设置GloVe模型
    glove = GloVe(n=n_dims, max_iter=60000, learning_rate=0.006)
    #获得GloVe模型的词向量
    wordEmbedding = glove.fit(wordComatrix) 
    print(token_id)
    print(wordEmbedding.shape)
    # 查询词向量
    print("抽搐的词向量为:", wordEmbedding[token_id['抽搐']])


# In[ ]:


token_id_mapping = token_id # 替换成自己的映射表
word_embedding_matrix = wordEmbedding # 替换成自己的词向量表
def preprocess_data(data,token_id_mapping):
    def preprocess_text(text):
        tokens = jieba.lcut(text)  # 使用jieba进行分词
        token_ids = [token_id_mapping.get(token, 0) for token in tokens]  # 将分词转换为映射表中的token_id
        token_ids_padded = token_ids[:20] + [0] * max(0, 20 - len(token_ids))  # 填充长度为50的序列
        return token_ids_padded

    text_list = data.tolist()  # 将指定文本列转换为文本列表
    token_ids_padded = [preprocess_text(text) for text in text_list]  # 对每个文本进行处理和转换
    return token_ids_padded

# 假设数据表中的主诉文本数据在 '主诉' 列中，使用的映射表为 token_id_mapping
token_ids_padded_np = preprocess_data(data_final['主诉'], token_id_mapping)
token_ids_padded_np = np.array(token_ids_padded_np, dtype=np.float64)

#合并文本ID列表数组和日期等数组
total_data = np.concatenate((token_ids_padded_np, data_encoder), axis=1)#(59774, 23),float32
print(total_data.shape)


# In[ ]:


Routings = 4 #5
Num_capsule = 5
Dim_capsule = 5#16
hidden_size = 50
word_size = 20
embed_size = 100
batch_size = 512 # how many samples to process at once
maxlen = 20 # max number of words in a question to use
embed_dim = 20

class Attention(tf.keras.layers.Layer):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        initializer = tf.keras.initializers.GlorotUniform()
        self.weight = self.add_weight(shape=(feature_dim, 1), initializer=initializer, trainable=True)
        if bias:
            self.b = self.add_weight(shape=(step_dim,), initializer="zeros", trainable=True)

    def call(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim
        eij = tf.matmul(tf.reshape(x, (-1, feature_dim)), self.weight)
        eij = tf.reshape(eij, (-1, step_dim))
        if self.bias:
            eij = eij + self.b
        eij = tf.tanh(eij)
        a = tf.exp(eij)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            a = a * mask
        a = a / (tf.reduce_sum(a, axis=1, keepdims=True) + 1e-10)
        weighted_input = x * tf.expand_dims(a, axis=-1)
        return tf.reduce_sum(weighted_input, axis=1)

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            'feature_dim': self.feature_dim,
            'step_dim': self.step_dim,
            'bias': self.bias
        })
        return config
    
class Caps_Layer(tf.keras.layers.Layer):
    def __init__(self, input_dim_capsule=hidden_size * 2, num_capsule=Num_capsule, dim_capsule=Dim_capsule,                  routings=Routings, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Caps_Layer, self).__init__(**kwargs)

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size  # 暂时没用到
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = tf.keras.activations.relu

        if self.share_weights:
            self.W = tf.Variable(
                tf.keras.initializers.glorot_normal()(shape=(1, hidden_size * 2, self.num_capsule * self.dim_capsule)),
                trainable=True)
        else:
            self.W = tf.Variable(
                tf.random.normal(shape=(batch_size, hidden_size * 2, self.num_capsule * self.dim_capsule)),
                trainable=True)

    def call(self, x):
        if self.share_weights:
            u_hat_vecs = tf.matmul(x, self.W)
        else:
            print('add later')

        batch_size = tf.shape(x)[0]
        input_num_capsule = tf.shape(x)[1]
        u_hat_vecs = tf.reshape(u_hat_vecs, (batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
        u_hat_vecs = tf.transpose(u_hat_vecs, perm=[0, 2, 1, 3])  # 转成(batch_size,num_capsule,input_num_capsule,dim_capsule)
        b = tf.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size,num_capsule,input_num_capsule)

        for i in range(self.routings):
            b = tf.transpose(b, perm=[0, 2, 1])
            c = tf.nn.softmax(b, axis=2)
            c = tf.transpose(c, perm=[0, 2, 1])
            b = tf.transpose(b, perm=[0, 2, 1])
            outputs = self.activation(tf.einsum('bij,bijk->bik', c, u_hat_vecs))  # batch matrix multiplication
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                b = tf.einsum('bik,bijk->bij', outputs, u_hat_vecs)  # batch matrix multiplication
        return outputs  # (batch_size, num_capsule, dim_capsule)

    # text version of squash, slight different from original one
    def squash(self, x, axis=-1):
        s_squared_norm = tf.reduce_sum(x ** 2, axis, keepdims=True)
        scale = tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
        return x / scale

    def get_config(self):
        config = super(Caps_Layer, self).get_config()
        config.update({
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings,
            'kernel_size': self.kernel_size,
            'share_weights': self.share_weights,
            'activation': self.activation
        })
        return config


class Capsule_Main(tf.keras.Model):
    def __init__(self, x, embedding_matrix=None, vocab_size=None):
        super(Capsule_Main, self).__init__()
        self.x = x 
        # self.embed_layer = Embed_Layer(embedding_matrix, vocab_size)
        self.gru_layer = GRU(x)
        self.caps_layer = Caps_Layer()
        self.dense_layer = tf.keras.layers.Dense(units=10)  # 指定输出单元的数量

    def call(self, content1):
        # content1 = self.embed_layer(content)
        content2, _ = self.gru_layer(content1)
        content3 = self.caps_layer(content2)
        output = self.dense_layer(content3)
        return output

    def get_config(self):
        config = super(Capsule_Main, self).get_config()
        config.update({
            'x': self.x,
            'embedding_matrix': self.embedding_matrix,
            'vocab_size': self.vocab_size
        })
        return config

def reshape_caps(x):
    shape = K.shape(x)
    return K.reshape(x, (shape[0], shape[1] * shape[2]))

class CustomEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, word_embedding_matrix, **kwargs):
        super(CustomEmbeddingLayer, self).__init__(**kwargs)
        self.embedding = Embedding(
            input_dim=word_embedding_matrix.shape[0],
            output_dim=word_embedding_matrix.shape[1],
            weights=[word_embedding_matrix],
            trainable=False
        )

    def call(self, inputs):
        return self.embedding(inputs)

def create_model1(word_embedding_matrix,maxlen, embed_dim, hidden_size,Num_capsule,Dim_capsule):
    total_input = Input(shape=(23,), name='total_input')
    text_input = total_input[:, :20]
    array_input = total_input[:, 20:]
    text_embedding = CustomEmbeddingLayer(word_embedding_matrix)(text_input)
    conv1 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu')(text_embedding)
    conv2 = Conv1D(filters = 64, kernel_size = 5, activation = 'relu')(text_embedding)
    conv3 = Conv1D(filters = 64, kernel_size = 7, activation = 'relu')(text_embedding)
    pool1 = GlobalMaxPooling1D()(conv1)
    pool2 = GlobalMaxPooling1D()(conv2)
    pool3 = GlobalMaxPooling1D()(conv3)
    conv_output = Concatenate()([pool1, pool2, pool3])
    x_lstm = Bidirectional(LSTM(units=hidden_size, return_sequences=True))(text_embedding)
    lstm_gru = Bidirectional(GRU(units=hidden_size, return_sequences=True))(x_lstm)
    
    lstm_att = Attention(hidden_size * 2, embed_dim)(x_lstm)
    gru_att = Attention(hidden_size * 2, embed_dim)(lstm_gru)
    
    avg_pool = GlobalAveragePooling1D()(lstm_gru)
    max_pool = GlobalMaxPooling1D()(lstm_gru)
    
    caps = Caps_Layer()(lstm_gru)
    caps_drop = Dropout(0.1)(caps)
#     batch_size = tf.shape(caps_drop)[0]
    caps_fla = Lambda(reshape_caps)(caps_drop)
    caps_out = Dense(units=5, activation='relu', input_shape=(25,))(caps_fla)
    
    
#     array_input = Input(shape=(3,),name='array_input') # 假设 x_figure 是一个长度为 3 的张量
    
#     conc = Concatenate()([lstm_att, gru_att, avg_pool, caps_out, max_pool, array_input])
    conc = Concatenate()([lstm_att, gru_att, caps_out,conv_output,array_input])
    
    conc = Dense(units=128, activation='relu')(conc)
    conc = Dense(units=64, activation='relu')(conc)
    conc = Dense(units=32, activation='relu')(conc)
    conc = BatchNormalization(axis=-1)(conc)
    conc = Dropout(0.1)(conc)
    out = Dense(units=15, activation='softmax')(conc)
    
    
    model = Model(inputs=total_input, outputs=out)
    model.summary()
    
    return model
model1 = create_model1(word_embedding_matrix,maxlen, embed_dim, hidden_size,Num_capsule,Dim_capsule)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(total_data, y, test_size=0.2, random_state=42)
X_train = tf.convert_to_tensor(X_train,dtype = tf.int64)
X_test = tf.convert_to_tensor(X_test,dtype = tf.int64)
y_train = tf.convert_to_tensor(y_train,dtype = tf.int64)
y_test = tf.convert_to_tensor(y_test,dtype = tf.int64)
print(X_train.dtype)
print(tf.shape(y_test))


# In[ ]:


# 编译模型
model1.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
model1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10,batch_size=64, steps_per_epoch=len(X_train) // 64,validation_steps=len(X_test) // 64)



loss_history = []
accuracy_history = []
precision_history = []
recall_history = []
f1_history = []

epochs = 10

# 训练模型并记录损失值和评估指标
for epoch in range(epochs):
    history = model1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=64, verbose=0)

    # 获取损失值和评估指标
    loss = history.history['loss'][0]
    accuracy = history.history['accuracy'][0]
    val_loss = history.history['val_loss'][0]
    val_accuracy = history.history['val_accuracy'][0]

    # 添加损失值和评估指标到列表中
    loss_history.append(loss)
    accuracy_history.append(accuracy)

    # 计算评估指标
    y_pred = model1.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    precision_history.append(precision)
    recall_history.append(recall)
    f1_history.append(f1)

# 绘制损失图
plt.figure()
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.ylim(-0.1, 1)  # 设置y轴的范围为0到1

# 绘制Accuracy图
plt.figure()
plt.plot(accuracy_history, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.ylim(-0.1, 1)  # 设置y轴的范围为0到1

# 绘制Precision图
plt.figure()
plt.plot(precision_history, label='Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision')
plt.legend()
plt.ylim(-0.1, 1)  # 设置y轴的范围为0到1

# 绘制Recall图
plt.figure()
plt.plot(recall_history, label='Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall')
plt.legend()
plt.ylim(-0.1, 1)  # 设置y轴的范围为0到1

plt.show()
    
# 保存模型
# model.save('mymodel', save_format='tf')

# 打印最终评估结果
print(f"Final Accuracy: {accuracy_history[-1]}")
print(f"Final Precision: {precision_history[-1]}")
print(f"Final Recall: {recall_history[-1]}")
print(f"Final F1 Score: {f1_history[-1]}")


# In[ ]:


X_train_np = np.array(X_train)
X_test_np = np.array(X_test)


# In[ ]:



explainer = shap.Explainer(model1, X_train_np)
shap_values = explainer(X_test_np[83:84])
# 获取 X_test_np[:1] 对应的特征值
max_prob_index = np.argmax(shap_values[0, :, :].sum(axis=1))
print(max_prob_index)
# 提取概率最大的结果的特征重要性
max_prob_shap_values = shap_values[0, :, max_prob_index]
print(max_prob_shap_values)
# 画出条形
shap.plots.bar(max_prob_shap_values)
# 展示图形
plt.show()


# In[ ]:


import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'SimSun'
print(X_test_np[83:84])
pred_probs = model1.predict(X_test_np[673:674])
max_prob_index = np.argmax(pred_probs)
print(max_prob_index)
feature_values = X_test_np[673:674].flatten().tolist()
num_features = X_test_np.shape[1]
# 构建完整的特征名称列表
feature_names = [f"feature{i+1}" for i in range(num_features)]
feature_labels = [f'{name}: {value}' for name, value in zip(feature_names, feature_values)]
# 创建一个用于计算SHAP值的KernelExplainer
ker_explainer= shap.KernelExplainer(model1.predict, X_train_np[:10])
shap_values_bar = ker_explainer.shap_values(X_test_np[673:674])
class_names = ['支气管炎','癫痫','支气管肺炎','上呼吸道感染','胃肠功能紊乱','中枢性性早熟','肾病综合征','抽动障碍','矮小症','扁桃体炎','急性肠炎','喘息性支气管炎','疱疹性咽峡炎','急性胃炎','手足口病']
plt.figure()
shap.summary_plot(shap_values_bar[max_prob_index], X_test_np[673:674], plot_type="bar", 
                  class_names = class_names,
                  max_display=10,feature_names=feature_labels,color = 'green', plot_size=(8, 3))

import seaborn as sns
# 在 shap.summary_plot() 后添加以下代码来显示结果类名称
pred_class_name = class_names[max_prob_index]
shap_values_class = shap_values_bar[max_prob_index]

# 将二维的 shap_values_class 转换为一维的数组
shap_values_1d = shap_values_class.flatten()

# 只保留前十个特征和对应的重要性
top_10_features = feature_labels[:10]
top_10_shap_values = shap_values_1d[:10]
print(feature_values)
# 将特征值转换为单词列表
word_features = [key for value in feature_values for key, val in token_id_mapping.items() if val == value]
word_features_top_10 = word_features[:10]

# 绘制带有类别标签的特征重要性图
plt.figure(figsize=(8, 3))
sns.barplot(x=top_10_shap_values, y=word_features_top_10, color='green')

# 添加分数标签
for i, value in enumerate(top_10_shap_values):
    plt.text(value, i, f'{value:.2f}', va='center')

plt.xlabel('SHAP Value')
plt.ylabel('Feature')
plt.title(f'Top 10 SHAP Values - Result Class: {pred_class_name}')
plt.show()


# In[ ]:


pred_probs = model1.predict(X_test_np[23:24])
print(pred_probs.shape)
# 找到每个样本的概率最大的类别索引
max_prob_index = np.argmax(pred_probs)
print(max_prob_index)
feature_values = X_test_np[:1].flatten().tolist()
num_features = X_test_np.shape[1]
# 构建完整的特征名称列表
feature_names = [f"feature{i+1}" for i in range(num_features)]
feature_labels = [f'{name}: {value}' for name, value in zip(feature_names, feature_values)]
# 创建一个用于计算SHAP值的KernelExplainer
ker_explainer= shap.KernelExplainer(model1.predict, X_train_np[23:24])
shap_values_bar = ker_explainer.shap_values(X_test_np[23:24])
print(tf.shape(shap_values_bar))
plt.figure()
shap.summary_plot(shap_values_bar, X_test_np[23:24], plot_type="bar", max_display=10,feature_names=feature_labels)
plt.show()


# In[ ]:


#dot
pred_probs_dot = model1.predict(X_test_np[23:24])
# 找到每个样本的概率最大的类别索引
max_prob_index_dot = np.argmax(pred_probs_dot)
print(max_prob_index_dot)

shap_values_dot = ker_explainer.shap_values(X_test_np[20:30])
print(tf.shape(shap_values_dot))
# 获取特征值
feature_names = [key for key, val in token_id.items()]

# 创建字典存储单词权重
word_weights = {}

# 遍历每个单词和对应的权重
for word, weight in zip(feature_names, shap_values_dot[2]):
    word_weights[word] = weight

# 打印每个单词的权重字典
# print(word_weights)
plt.figure()
shap.summary_plot(shap_values_dot[2], X_test_np[20:30], plot_type="dot", max_display=10)
plt.show()


# In[ ]:


#force
shap.initjs()
#单个例子,也可多个（难理解）
pred_probs_for = model1.predict(X_test_np[73:74])
# 找到概率最大的类别索引
max_prob_index_for = np.argmax(pred_probs_for)
print(max_prob_index_for)
shap_values_force = ker_explainer.shap_values(X_test_np[73:74])
print(tf.shape(shap_values_force))



feature_values = X_test_np[73:74].flatten().tolist()
class_names = ['支气管炎','癫痫','支气管肺炎','上呼吸道感染','胃肠功能紊乱','中枢性性早熟','肾病综合征','抽动障碍','矮小症','扁桃体炎','急性肠炎','喘息性支气管炎','疱疹性咽峡炎','急性胃炎','手足口病']

# 将特征值转换为单词列表
word_features = [key for value in feature_values for key, val in token_id_mapping.items() if val == value]
print(word_features)
# 创建 SHAP 值图
plt.figure(figsize=(8, 4))  # 调整图的大小
shap.force_plot(ker_explainer.expected_value[max_prob_index_for], shap_values_force[max_prob_index_for],
                feature_names=word_features, matplotlib=True, show=False)

# 添加文本
# plt.text(0.5, -0.4, "鼻塞、流涕3天，咳嗽2天", fontsize=20, ha='center',color = (0.23, 0.40, 1))

# 显示图形
plt.show()


# In[ ]:


#决策图
pred_probs_dec = model1.predict(X_test_np[:1])
max_prob_index_dec = np.argmax(pred_probs_dec)
shap_values_dec = ker_explainer.shap_values(X_test_np[:1])
shap.decision_plot(ker_explainer.expected_value[max_prob_index_dec], shap_values_dec[max_prob_index_dec],feature_names=word_features,)


# In[ ]:


# 模型预测
pred_probs_dot = model1.predict(X_test_np[73:74])
# 找到概率最大的类别索引
max_prob_index_dot = np.argmax(pred_probs_dot)
print(max_prob_index_dot)
# 获取SHAP值
shap_values_dot = ker_explainer.shap_values(X_test_np[73:74])
print(tf.shape(shap_values_dot))
# 获取特征名和对应的SHAP值
feature_names = [key for key, val in token_id.items()]
shap_values = shap_values_dot
# 计算所有单词贡献程度的绝对值之和
total_contributions = sum(abs(contribution) for _, contribution in word_contributions.items())

# 创建字典存储归一化后的单词贡献程度
normalized_contributions = {}

# 遍历每个单词和对应的贡献程度
for word, contribution in word_contributions.items():
    normalized_contributions[word] = contribution / total_contributions

# 按归一化后的贡献程度由大到小排序
sorted_normalized_contributions = sorted(normalized_contributions.items(), key=lambda x: abs(x[1]), reverse=True)

# 打印每个单词的归一化后的贡献程度
for word, contribution in sorted_normalized_contributions:
    print(f"单词 '{word}' 对分类结果的归一化贡献程度: {contribution}")

# 创建 SHAP 值图
plt.figure()
shap.summary_plot(shap_values_dot, X_test_np[73:74], plot_type="bar", max_display=10)
plt.show()

