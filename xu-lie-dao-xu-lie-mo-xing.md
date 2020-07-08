# 序列到序列模型

序列到序列模型在自然语言处理中应用广泛，是重要的模型结构。本小节对序列到序列模型的提出和结构进行简要介绍，没有涉及代码实现部分。

## 提出问题

前面章节讲到的RNN模型和实例，都属于序列预测问题，或是通过序列中一个时间步的输入值，预测下一个时间步输出值（如二进制减法问题）；或是对所有输入序列得到一个输出作为分类（如名字分类问题）。他们的共同特点是：输出序列与输入序列等长，或输出长度为1。

还有一类序列预测问题，以序列作为输入，需要输出也是序列，并且输入和输出序列长度不确定，并不断变化。这类问题被成为序列到序列（Sequence-to-Sequence, Seq2Seq）预测问题。

序列到序列问题有很多应用场景：比如机器翻译、问答系统（QA）、文档摘要生成等。简单的RNN或LSRM结构无法处理这类问题，于是科学家们提出了一种新的结构 —— 编码解码（Encoder-Decoder）结构。

## 编码-解码结构（Encoder-Decoder）

图20-9 为Encoder-Decoder结构的示意图。

![&#x56FE;20-9 Encoder-Decoder&#x7ED3;&#x6784;&#x793A;&#x610F;&#x56FE;](.gitbook/assets/image%20%2816%29.png)

Encoder-Decoder结构的处理流程非常简单直观。

* 示意图中，输入序列和输出序列分别为中文语句和翻译之后的英文语句，它们的长度不一定相同。通常会将输入序列嵌入（Embedding）成一定维度的向量，传入编码器。
* Encoder为编码器，将输入序列编码成为固定长度的状态向量，通常称为语义编码向量。
* Decoder为解码器，将语义编码向量作为原始输入，解码成所需要的输出序列。

在具体实现中，编码器、解码器可以有不同选择，可自由组合。常见的选择有CNN、RNN、GRU、LSTM等。

应用Encoder-Decoder结构，可构建出序列到序列模型。

## 序列到序列模型（Seq2Seq）

Seq2Seq模型有两种常见结构。我们以RNN网络作为编码和解码器来进行讲解。

图20-10和图20-11分别展示了这两种结构。

![&#x56FE;20-10 Seq2Seq&#x7ED3;&#x6784;&#x4E00;](.gitbook/assets/image%20%287%29.png)

![&#x56FE;20-11 Seq2Seq&#x7ED3;&#x6784;&#x4E8C;](.gitbook/assets/image%20%2820%29.png)

### 编码过程

两种结构的编码过程完全一致。

输入序列为 $$x=[x1, x2, x3]$$。

RNN网络中，每个时间节点隐层状态为:

$$ h_t = f(h_{t-1}, x_t), \quad t \in [1,3] $$

编码器中输出的语义编码向量可以有三种不同选取方式，分别是：

$$
\begin{aligned}
c &= h_3 \\
c &= g(h_3) \\
c &= g(h1, h2, h3) \\
\end{aligned}
$$

### 解码过程

两种结构解码过程的不同点在于，语义编码向量是否应用于每一时刻输入。

第一种结构，每一时刻的输出$$y_t$$由前一时刻的输出$$y_{t-1}$$、前一时刻的隐层状态$$h^{'}{t-1}$$_和_$$c$$共同决定，即_：_ $$y_t = f(y{t-1}, h^{'}_{t-1}, c)$$。

第二种结构，$$c$$只作为初始状态传入解码器，并不参与每一时刻的输入，即：

$$\begin{cases}     y_1 = f(y_0, h^{'}_{0}, c) \\     y_t = f(y_{t-1}, h^{'}_{t-1}), t \in [2,4] \end{cases}$$

以上是序列到序列模型的结构简介。

## keras实战

这里我们使用一个机器翻译的例子来进行实战讲解。

我们希望可以通过模型实现机器翻译，例如英语转法语：

```text
"the cat sat on the mat" -> [Seq2Seq model] -> "le chat etait assis sur le tapis"
```

在一般情况下，输入序列和输出序列具有不同的长度\(例如机器翻译\) ，为了开始预测目标，需要整个输入序列。对于我们的示例实现，我们将使用英语句子对及其法语翻译的数据集，您可以从 [manythings.org/anki ](http://www.manythings.org/anki/)下载。要下载的文件名为 fra-eng.zip。我们将实现一个字符级的序列到序列模型，逐个字符处理输入并逐个字符生成输出。另一种选择是词级模型，这种模型在机器翻译中更为常见。

## 算法流程

以下是我们整个过程的总结:

* 1\) 把句子转换成3个 Numpy 数组,`encoder_input_data`, `decoder_input_data`, `decoder_target_data`:
  * `encoder_input_data`是一个三维数组`(num_pairs, max_english_sentence_length, num_english_characters)` 包含英语句子的one-hot编码。
  * `decoder_input_data` 是一个三维数组`(num_pairs, max_french_sentence_length, num_french_characters)` 包含法语句子的one-hot编码
  * `decoder_target_data`与`decoder_input_data`相同，但偏移了一个时间步长。`decoder_target_data[:， t，:]`将与`decoder_input_data[:， t + 1，:]`相同。
* 2\) 训练一个基本的基于lstm的Seq2Seq模型来预测给定`encoder_input_data`和`decoder_input_data`的`decoder_target_data`。
* 3\) 解码一些句子来检查模型是否正常工作\(例如，将样本从`encoder_input_data`转换为相应的样本`decoder_target_data`\)。

因为训练过程和推理过程\(解码句子\)是完全不同的，所以我们对两者使用不同的模型，尽管它们都利用了相同的内部层次。

### 完整代码

```python
from __future__ import print_function
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = '../data/fra-eng/fra.txt'

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
```

## 参考文献

\[1\] keras-example: [lstm\_seq2seq](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)

