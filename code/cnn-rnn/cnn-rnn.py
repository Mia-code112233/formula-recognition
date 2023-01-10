#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import *
import os
import tqdm
import numpy as np


label_dir = "label/"
image_dir = "image/"

"""
training_img = np.load('autodl-nas/img2latex/img2latex_cache/training_img.npy').tolist()
validate_img = np.load('autodl-nas/img2latex/img2latex_cache/validate_img.npy').tolist()
"""

# # Part 1. 提取图片特征值
# * training_features_path: 使用inception_resnet提取出来的训练集图片特征
# * validate_features_path: 使用inception_resnet提取出来的验证集图片特征
# * label_dir：使用老师给的脚本进行预处理后的latex文件目录
# * image_dir: 根据latex文件寻得的具有标签的图片目录

# In[2]:


label_name_list = os.listdir(label_dir)
label_name_list = label_name_list[:500]


# In[3]:


training_labels = []
# 建立label<->image的映射
# key:label_name, value:image path
label_to_image = {} 
# key:image name ("xxx.png"), value:latex with <start> <end>
image_to_label = {}
for label_name in tqdm.tqdm(label_name_list):
    label_to_image[label_name] = image_dir + label_name[:-4] + ".png"
    with open(label_dir + label_name, 'r', encoding='UTF-8') as f:
        tokenized_latex = f.read()
        tokenized_latex = f"<start> {tokenized_latex} <end>"
    image_to_label[label_name[:-4]+".png"] = tokenized_latex


# In[4]:


# creating training set and validation set (8:2)

img_keys = list(image_to_label.keys())

import random
random.shuffle(img_keys)
slice_index = int(len(img_keys)*0.8)

training_img, validate_img = img_keys[:slice_index], img_keys[slice_index:]



# # 使用高层CNN接口来对图片进行特征提取（v1）
# * 在v1中，使用了inception_resnet_v2来进行特征提取，看看效果怎么样
# * 后续可以改用inception_v3，resnet等
# 
# ## load_image
# * 使用tensorflow内置的函数根据图片路径载入图片，并resize为(299, 299),即inception_resnet的输入大小
# * 调用preprocess_input对其进行预处理
# * load_image后续是供map使用的
# 
# ### feature_extract_model:用于提取图片特征的cnn模型
# * weights使用imagenet初始化

# In[5]:



cnn_model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False)
# cnn_model = applications.inception_v3.InceptionV3(include_top=False, weights=None)

new_input = cnn_model.input
new_output = cnn_model.layers[-1].output

feature_extract_model = Model(new_input, new_output)


# # Part II. 根据提取的特征值来构建数据集并训练
# * 上方的代码仅仅执行一次就可以了

# # 根据提取特征文件夹中的文件创建tokenizer
# * 即创建训练图片对应的sequence

# In[2]:




training_target = []
validate_target = []

for feature in tqdm.tqdm(training_img):
    target_path = label_dir + feature[:-4] + ".txt"
    with open(target_path, 'r', encoding='UTF-8') as f:
        tokenized_latex = f.read()
        tokenized_latex = f"<start> {tokenized_latex} <end>"
        training_target.append(tokenized_latex)

for feature in tqdm.tqdm(validate_img):
    validate_path = label_dir + feature[:-4] + ".txt"
    with open(validate_path, 'r', encoding='UTF-8') as f:
        tokenized_latex = f.read()
        tokenized_latex = f"<start> {tokenized_latex} <end>"
        validate_target.append(tokenized_latex)


# In[3]:


# trun sentence into sequence
tokenizer = preprocessing.text.Tokenizer(filters="", oov_token='<unk>')
tokenizer.index_word[0] = '<pad>'
tokenizer.word_index['<pad>'] = 0
tokenizer.fit_on_texts(training_target)
sequences = tokenizer.texts_to_sequences(training_target)
target_vector = preprocessing.sequence.pad_sequences(sequences, padding='post')

print(len(target_vector[0]))
max_len = len(target_vector[0])


# In[12]:


tokenizer.fit_on_texts(validate_target)
validate_sequences = tokenizer.texts_to_sequences(validate_target)
validate_vector = preprocessing.sequence.pad_sequences(validate_sequences, padding='post')

word_dict = eval(tokenizer.get_config()['word_index'])
word_dict_len = len(word_dict.keys())

validate_max_len = len(validate_vector[0])
tokenizer.index_word[0] = '<pad>'
tokenizer.word_index['<pad>'] = 0


# In[ ]:


print(tokenizer.index_word)


# # 创建用于训练的dataset
# **在提取完特征后，图片dataset都基于提取的特征，training_latex和validate_latex等都需要重新构造**
# * 输入是加载的.npy features(保存为npy的float32)
# * 输出是图片对应的latex(已经map为sequence的int32)

# In[12]:


BUFFER_SIZE = 10000
BATCH_SIZE = 64
VOCAB_SIZE = word_dict_len + 1
DENSE_OUTPUT_SIZE = 256
EMBEDDING_DIM = 128
UNITS = 256

def load_image(image_name, latex):
    image_path = image_dir + image_name
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = applications.inception_resnet_v2.preprocess_input(img)
    return img, latex

dataset = tf.data.Dataset.from_tensor_slices((training_img, target_vector))
print(dataset)
dataset = dataset.map(load_image)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

print(dataset)


# In[13]:


validate_dataset = tf.data.Dataset.from_tensor_slices((validate_img, validate_vector))
validate_dataset = validate_dataset.map(load_image)

validate_dataset = validate_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
validate_dataset = validate_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

print(validate_dataset)


# # 构造模型
# * cnn：对之前提取的特征再做一次全连接
# * attention：使用的是BahdanauAttention模型
# * rnn：rnn-cell使用的是gru

# In[14]:


class Encoder(Model):
    def __init__(self, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.dense = layers.Dense(units=DENSE_OUTPUT_SIZE,activation='relu')
    
    def call(self, x):
        x = self.dense(x)
        return x

class Attention(Model):
    def __init__(self, units=UNITS, *args, **kwargs):
        super(Attention, self).__init__(*args, **kwargs)
        self.dense1 = layers.Dense(units)
        self.dense2 = layers.Dense(units)
        self.out = layers.Dense(1)

    def call(self, features, hidden):

        hidden_with_time = tf.expand_dims(hidden, axis=1)

        attention_hidden_layer = tf.nn.tanh(self.dense1(features)+self.dense2(hidden_with_time))

        result = self.out(attention_hidden_layer)
        
        weights = tf.nn.softmax(result, axis=1)

        context_vector = weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, weights
        

class Decoder(Model):
    def __init__(self, embedding_dim=EMBEDDING_DIM, units=UNITS, vocab_size=VOCAB_SIZE, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.gru = layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.dense1 = layers.Dense(self.units)
        self.dense2 = layers.Dense(vocab_size)

        self.attention = Attention(self.units)

    def call(self, x, features, hidden):
        context, weights = self.attention(features, hidden)
        # x = (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x = (batch_size, 1, embedding_dim+hidden_size)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        output, state = self.gru(x)
        # x = (batch_size, max_sequence, hiddensize)
        # max_sequence is the length of the longest token sequence
        x = self.dense1(output)
        # x = (batch_size * max_sequence)
        x = tf.reshape(x, [-1, x.shape[2]])

        # x = (batch_size * max_sequence, vocab_size)
        x = self.dense2(x)

        return x, state, weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

optimizer = optimizers.Adam(learning_rate=0.008)
loss_object = losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
  loss_ = loss_object(real, pred)

  return tf.reduce_mean(loss_)
        


# # 创建训练检查点
# * 防止jupyter中间出现问题
# * 之后还会使用saved_model来保存模型

# In[15]:


encoder = Encoder()
decoder = Decoder()


checkpoint_path = "checkpoint/"
ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)


# # 训练模型
# * 定义train_step
# * 使用了teaching forcing来喂输入
#     * 后续可以加入和beam search, free的对比
# 
# * 需要用categoriacal来sample原始的prediction（理解一下）

# In[16]:


def exact_match(y_pred, y_true):
    match = tf.math.equal(y_pred, y_true)
    match = tf.reduce_min(tf.cast(match, tf.float32), axis=1)
    
    return tf.reduce_mean(match)

def edit_distance(y_pred, y_true):
    ldistance = 1-tf.edit_distance(y_pred, y_true)
    return tf.reduce_mean(ldistance)

def saver():
  np.save('cache/loss_plot.npy', np.array(loss_plot))
  np.save('cache/exact_match_plot.npy', np.array(exact_match_plot))
  np.save('cache/edit_distance_plot.npy', np.array(edit_distance_plot))

  np.save('cache/validate_loss_plot.npy', np.array(validate_loss_plot))
  np.save('cache/validate_exact_match_plot.npy', np.array(validate_exact_match_plot))
  np.save('cache/validate_edit_distance_plot.npy', np.array(validate_edit_distance_plot))

"""
loss_plot = np.load('F:\img2latex\cache\\loss_plot.npy').tolist()
exact_match_plot = np.load('F:\img2latex\cache\\exact_match_plot.npy').tolist()
edit_distance_plot = np.load('F:\img2latex\cache\\edit_distance_plot.npy').tolist()
validate_loss_plot = np.load('F:\img2latex\cache\\validate_loss_plot.npy').tolist()
validate_exact_match_plot = np.load('F:\img2latex\cache\\validate_exact_match_plot.npy').tolist()
validate_edit_distance_plot = np.load('F:\img2latex\cache\\validate_edit_distance_plot.npy').tolist()
"""

loss_plot = []
exact_match_plot = []
edit_distance_plot = []
validate_loss_plot = []
validate_exact_match_plot = []
validate_edit_distance_plot = []


# In[ ]:


start_epoch = len(loss_plot)

def train_step(img, target):
    loss = 0

    # initial state according to batch
    hidden = decoder.reset_state(target.shape[0])

    # initial batch_size nums of <start>
    decoder_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    decoder_output = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
    decoder_output = tf.cast(decoder_output, tf.float32)
    img_tensor = feature_extract_model(img)
    img_tensor = tf.reshape(img_tensor, [img_tensor.shape[0], -1, img_tensor.shape[3]])
    
    mask = layers.Masking()
    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            predictions, hidden, _ = decoder(decoder_input, features, hidden)

            loss += loss_function(target[:,i], predictions)

            categorical_output = tf.random.categorical(predictions, 1, dtype=tf.int32)
            categorical_output = tf.cast(categorical_output, tf.float32)
            
           
            categorical_mask = mask(tf.expand_dims(tf.cast(target[:,i], tf.float32), -1))._keras_mask
            categorical_mask = tf.transpose(categorical_mask)
            
            categorical_mask = tf.expand_dims(tf.cast(categorical_mask,tf.float32), axis=1)
            categorical_output *= categorical_mask
            decoder_input = categorical_output

            decoder_output = tf.concat([decoder_output, categorical_output], 1)
            
            # teacher forcing
            decoder_input = tf.expand_dims(target[:,i], 1)

    decoder_output = tf.cast(decoder_output, tf.int32)

    batch_edit_distance = edit_distance(tf.sparse.from_dense(decoder_output), tf.sparse.from_dense(target))

    batch_exact_match = exact_match(decoder_output, target)

    total_loss = loss / int(target.shape[1])

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss, batch_exact_match, batch_edit_distance

def validate_step(img, target):
    loss = 0
    
    hidden = decoder.reset_state(target.shape[0])
    decoder_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    decoder_output = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
    decoder_output = tf.cast(decoder_output, tf.float32)

    img_tensor = feature_extract_model(img)
    img_tensor = tf.reshape(img_tensor, [img_tensor.shape[0], -1, img_tensor.shape[3]])
    features = encoder(img_tensor)
    mask = layers.Masking()
    for i in range(1, target.shape[1]):
        predictions, hidden, _ = decoder(decoder_input, features, hidden)

        loss += loss_function(target[:,i], predictions)

        categorical_output = tf.random.categorical(predictions, 1)
        categorical_output = tf.cast(categorical_output, tf.float32)
        
        categorical_mask = mask(tf.expand_dims(tf.cast(target[:,i], tf.float32), -1))._keras_mask
        categorical_mask = tf.transpose(tf.cast(categorical_mask, tf.float32))
        categorical_mask = tf.expand_dims(categorical_mask, axis=1)
        
        categorical_output *= categorical_mask
        decoder_output = tf.concat([decoder_output, categorical_output], 1)

        decoder_input = categorical_output
    

    decoder_output = tf.cast(decoder_output, tf.int32)

    batch_edit_distance = edit_distance(tf.sparse.from_dense(decoder_output), tf.sparse.from_dense(target))
    
    batch_exact_match = exact_match(decoder_output, target)

    total_loss = loss / int(target.shape[1])

    return loss, total_loss, batch_exact_match, batch_edit_distance


EPOCHS = 20
num_steps = len(training_img) // BATCH_SIZE
validate_num_steps = len(validate_img) // BATCH_SIZE

for epoch in range(start_epoch, EPOCHS):

    # training and update the weights
    total_loss = 0
    epoch_exact_match = 0
    epoch_edit_distance = 0
    for (batch, (img, target)) in enumerate(dataset):
        batch_loss, t_loss, ematch, edistance = train_step(img, target)
        total_loss += t_loss
        
        epoch_exact_match += ematch
        epoch_edit_distance += edistance
        
        if batch % 100 == 0:
          average_batch_loss = batch_loss.numpy()/int(target.shape[1])
          print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f} ')
    
    # get the validate result
    validate_total_loss = 0
    validate_epoch_exact_match = 0
    validate_epoch_edit_distance = 0
    for(batch, (img, target)) in enumerate(validate_dataset):
        batch_loss, t_loss, ematch, edistance = validate_step(img, target)
        validate_total_loss += t_loss

        validate_epoch_exact_match += ematch
        validate_epoch_edit_distance += edistance
        if batch % 100 == 0:
          average_batch_loss = batch_loss.numpy()/int(target.shape[1])
          print(f'[validate] : Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f} ')
    


    exact_match_plot.append(epoch_exact_match/num_steps)
    edit_distance_plot.append(epoch_edit_distance/num_steps)
    loss_plot.append(total_loss/num_steps)

    validate_loss_plot.append(validate_total_loss / validate_num_steps)
    validate_exact_match_plot.append(validate_epoch_exact_match / validate_num_steps)
    validate_edit_distance_plot.append(validate_epoch_edit_distance / validate_num_steps)
    manager.save()
    saver()
    print(f'Epoch {epoch+1} Loss {total_loss/num_steps:.6f} Exact Match {epoch_exact_match/num_steps:.6f} EditDistance {epoch_edit_distance/num_steps:.6f}')
    print(f'[validate]: Epoch {epoch+1} Loss {validate_total_loss/validate_num_steps:.6f} Exact Match {validate_epoch_exact_match/validate_num_steps:.6f} EditDistance {validate_epoch_edit_distance/validate_num_steps:.6f}')


# # 作图

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
x_axis = range(EPOCHS)

fig = plt.figure(figsize=(11,11))
plt.subplot(3, 1, 1)
plt.plot(x_axis, loss_plot, color='r', label='training loss')
plt.plot(x_axis, validate_loss_plot, color='b', label='validate loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('metrics')
plt.title('loss')


plt.subplot(3, 1, 2)
plt.plot(x_axis, edit_distance_plot, color='r', label='training edit distance')
plt.plot(x_axis, validate_edit_distance_plot, color='b', label='validate edit distance')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('metrics')
plt.title('edit distance')

plt.subplot(3, 1, 3)
plt.plot(x_axis, exact_match_plot, color='r', label='training exact match')
plt.plot(x_axis, validate_exact_match_plot, color='b', label='validate exact match')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('metrics')
plt.title('exact match')


plt.savefig("cache/result.png")


# In[ ]:



# In[ ]:




