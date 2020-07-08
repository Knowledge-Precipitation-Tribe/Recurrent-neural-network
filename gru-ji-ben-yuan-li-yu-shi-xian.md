# GRU基本原理与实现

## GRU 的基本概念

LSTM 存在很多变体，其中门控循环单元（Gated Recurrent Unit, GRU）是最常见的一种，也是目前比较流行的一种。GRU是由 [Cho](https://arxiv.org/pdf/1406.1078v3.pdf) 等人在2014年提出的，它对LSTM做了一些简化：

1. GRU将LSTM原来的三个门简化成为两个：重置门 $$r_t$$（Reset Gate）和更新门 $$z_t$$ \(Update Gate\)。
2. GRU不保留单元状态 $$c_t$$，只保留隐藏状态 $$h_t$$作为单元输出，这样就和传统RNN的结构保持一致。
3. 重置门直接作用于前一时刻的隐藏状态 $$h_{t-1}$$。

## GRU的前向计算

### GRU的单元结构

图20-7展示了GRU的单元结构。

![&#x56FE;20-7 GRU&#x5355;&#x5143;&#x7ED3;&#x6784;&#x56FE;](.gitbook/assets/image%20%286%29.png)

GRU单元的前向计算公式如下：

1. 更新门

   $$ z_t = \sigma(h_{t-1} \cdot W_z + x_t \cdot U_z) \tag{1} $$

2. 重置门

   $$ r_t = \sigma(h_{t-1} \cdot W_r + x_t \cdot U_r) \tag{2} $$

3. 候选隐藏状态

   $$ \tilde{h}t = \tanh((r_t \circ h{t-1}) \cdot W_h + x_t \cdot U_h) \tag{3} $$

4. 隐藏状态

   $$ h = (1 - z_t) \circ h_{t-1} + z_t \circ \tilde{h}_t \tag{4} $$

### GRU的原理浅析

从上面的公式可以看出，GRU通过更新们和重置门控制长期状态的遗忘和保留，以及当前输入信息的选择。更新门和重置门通过$$sigmoid$$函数，将输入信息映射到$$[0,1]$$区间，实现门控功能。

首先，上一时刻的状态$$h_{t-1}$$通过重置门，加上当前时刻输入信息，共同构成当前时刻的即时状态$$\tilde{h}_t$$，并通过$$\tanh$$函数映射到$$[-1,1]$$区间。

然后，通过更新门实现遗忘和记忆两个部分。从隐藏状态的公式可以看出，通过$$z_t$$进行选择性的遗忘和记忆。$$(1-z_t)$$和$$z_t$$有联动关系，上一时刻信息遗忘的越多，当前信息记住的就越多，实现了LSTM中$$f_t$$和$$i_t$$的功能。

## GRU的反向传播

学习了LSTM的反向传播的推导，GRU的推导就相对简单了。我们仍然以$$l$$层$$t$$时刻的GRU单元为例，推导反向传播过程。

同LSTM， 令：$$l$$层$$t$$时刻传入误差为$$\delta_{t}^l$$，为下一时刻传入误差$$\delta_{h_t}^l$$和上一层传入误差$$\delta_{x_t}^{l+1}$$之和，简写为$$\delta_{t}$$。

令：

$$ z_{zt} = h_{t-1} \cdot W_z + x_t \cdot U_z \tag{5} $$

$$ z_{rt} = h_{t-1} \cdot W_r + x_t \cdot U_r \tag{6} $$

$$ z_{\tilde{h}t} = (r_t \circ h{t-1}) \cdot W_h + x_t \cdot U_h \tag{7} $$

则：

$$ \begin{aligned} \delta_{z_{zt}} = \frac{\partial{loss}}{\partial{h_t}} \cdot \frac{\partial{h_t}}{\partial{z_t}} \cdot \frac{\partial{z_t}}{\partial{z_{z_t}}} \\ = \delta_t \cdot (-diag[h_{t-1}] + diag[\tilde{h}_t]) \cdot diag[z_t \circ (1-z_t)] \\ = \delta_t \circ (\tilde{h}t - h{t-1}) \circ z_t \circ (1-z_t) \end{aligned} \tag{8} $$

$$ \begin{aligned} \delta_{z_{\tilde{h}t}} = \frac{\partial{loss}}{\partial{h_t}} \cdot \frac{\partial{h_t}}{\partial{\tilde{h}_t}} \cdot \frac{\partial{\tilde{h}t}}{\partial{z{\tilde{h}_t}}} \ = \delta_t \cdot diag[z_t] \cdot diag[1-(\tilde{h}_t)^2] \ &= \delta_t \circ z_t \circ (1-(\tilde{h}_t)^2) \end{aligned} \tag{9} $$$$ \begin{aligned} \delta_{z_{rt}} = \frac{\partial{loss}}{\partial{\tilde{h}t}} \cdot \frac{\partial{\tilde{h}t}}{\partial{z{\tilde{h}t}}} \cdot \frac{\partial{z{\tilde{h}t}}}{\partial{r_t}} \cdot \frac{\partial{r_t}}{\partial{z{r_t}}} \\ = \delta{z_{\tilde{h}t}} \cdot W_h^T \cdot diag[h_{t-1}] \cdot diag[r_t \circ (1-r_t)] \ &= \delta_{z_{\tilde{h}t}} \cdot W_h^T \circ h_{t-1} \circ r_t \circ (1-r_t) \end{aligned} \tag{10} $$

由此可求出，$$t$$时刻各个可学习参数的误差：

$$ \begin{aligned} d_{W_{h,t}} = \frac{\partial{loss}}{\partial{z_{\tilde{h}t}}} \cdot \frac{\partial{z{\tilde{h}t}}}{\partial{W_h}} = (r_t \circ h{t-1})^T \cdot \delta_{z_{\tilde{h}t}} \end{aligned} \tag{11} $$

$$ \begin{aligned} d_{U_{h,t}} = \frac{\partial{loss}}{\partial{z_{\tilde{h}t}}} \cdot \frac{\partial{z{\tilde{h}t}}}{\partial{U_h}} = x_t^T \cdot \delta{z_{\tilde{h}t}} \end{aligned} \tag{12} $$

$$ \begin{aligned} d_{W_{r,t}} = \frac{\partial{loss}}{\partial{z_{r_t}}} \cdot \frac{\partial{z_{r_t}}}{\partial{W_r}} = h_{t-1}^T \cdot \delta_{z_{rt}} \end{aligned} \tag{13} $$

$$ \begin{aligned} d_{U_{r,t}} = \frac{\partial{loss}}{\partial{z_{r_t}}} \cdot \frac{\partial{z_{r_t}}}{\partial{U_r}} = x_t^T \cdot \delta_{z_{rt}} \end{aligned} \tag{14} $$

$$ \begin{aligned} d_{W_{z,t}} = \frac{\partial{loss}}{\partial{z_{z_t}}} \cdot \frac{\partial{z_{z_t}}}{\partial{W_z}} = h_{t-1}^T \cdot \delta_{z_{zt}} \end{aligned} \tag{15} $$

$$ \begin{aligned} d_{U_{z,t}} = \frac{\partial{loss}}{\partial{z_{z_t}}} \cdot \frac{\partial{z_{z_t}}}{\partial{U_z}} = x_t^T \cdot \delta_{z_{zt}} \end{aligned} \tag{16} $$

可学习参数的最终误差为各个时刻误差之和，即：

$$ d_{W_h} = \sum_{t=1}^{\tau} d_{W_{h,t}} = \sum_{t=1}^{\tau} (r_t \circ h_{t-1})^T \cdot \delta_{z_{\tilde{h}t}} \tag{17} $$

$$ d_{U_h} = \sum_{t=1}^{\tau} d_{U_{h,t}} = \sum_{t=1}^{\tau} x_t^T \cdot \delta_{z_{\tilde{h}t}} \tag{18} $$

$$ d_{W_r} = \sum_{t=1}^{\tau} d_{W_{r,t}} = \sum_{t=1}^{\tau} h_{t-1}^T \cdot \delta_{z_{rt}} \tag{19} $$

$$ d_{U_r} = \sum_{t=1}^{\tau} d_{U_{r,t}} = \sum_{t=1}^{\tau} x_t^T \cdot \delta_{z_{rt}} \tag{20} $$

$$ d_{W_z} = \sum_{t=1}^{\tau} d_{W_{z,t}} = \sum_{t=1}^{\tau} h_{t-1}^T \cdot \delta_{z_{zt}} \tag{21} $$

$$ d_{U_z} = \sum_{t=1}^{\tau} d_{U_{z,t}} = \sum_{t=1}^{\tau} x_t^T \cdot \delta_{z_{zt}} \tag{22} $$

当前GRU cell分别向前一时刻（$t-1$）和下一层（$l-1$）传递误差，公式如下：

沿时间向前传递：

$$
\begin{aligned} \delta_{h_{t-1}} = \frac{\partial{loss}}{\partial{h_{t-1}}} \\ = \frac{\partial{loss}}{\partial{h_t}} \cdot \frac{\partial{h_t}}{\partial{h_{t-1}}} + \frac{\partial{loss}}{\partial{z_{\tilde{h}t}}} \cdot \frac{\partial{z{\tilde{h}t}}}{\partial{h{t-1}}} \ &+ \frac{\partial{loss}}{\partial{z_{rt}}} \cdot \frac{\partial{z_{rt}}}{\partial{h_{t-1}}} + \frac{\partial{loss}}{\partial{z_{zt}}} \cdot \frac{\partial{z_{zt}}}{\partial{h_{t-1}}} \\ = \delta_{t} \circ (1-z_t) + \delta_{z_{\tilde{h}t}} \cdot W_h^T \circ r_t \ &+ \delta_{z_{rt}} \cdot W_r^T + \delta_{z_{zt}} \cdot W_z^T \end{aligned} \tag{23}
$$

沿层次向下传递：

$$ \begin{aligned} \delta_{x_t} = \frac{\partial{loss}}{\partial{x_t}} = \frac{\partial{loss}}{\partial{z_{\tilde{h}t}}} \cdot \frac{\partial{z{\tilde{h}t}}}{\partial{x_t}} \ &+ \frac{\partial{loss}}{\partial{z{r_t}}} \cdot \frac{\partial{z_{r_t}}}{\partial{x_t}} + \frac{\partial{loss}}{\partial{z_{z_t}}} \cdot \frac{\partial{z_{z_t}}}{\partial{x_t}} \\ = \delta_{z_{\tilde{h}t}} \cdot U_h^T + \delta_{z_{rt}} \cdot U_r^T + \delta_{z_{zt}} \cdot U_z^T \end{aligned} \tag{24} $$

以上，GRU反向传播公式推导完毕。

## 代码实现

本节进行了GRU网络单元前向计算和反向传播的实现。为了统一和简单，测试用例依然是二进制减法。

### 初始化

本案例实现了没有bias的GRU单元，只需初始化输入维度和隐层维度。

```python
def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
```

### 前向计算

```python
def forward(self, x, h_p, W, U):
    self.get_params(W, U)
    self.x = x

    self.z = Sigmoid().forward(np.dot(h_p, self.wz) + np.dot(x, self.uz))
    self.r = Sigmoid().forward(np.dot(h_p, self.wr) + np.dot(x, self.ur))
    self.n = Tanh().forward(np.dot((self.r * h_p), self.wn) + np.dot(x, self.un))
    self.h = (1 - self.z) * h_p + self.z * self.n

def split_params(self, w, size):
        s=[]
        for i in range(3):
            s.append(w[(i*size):((i+1)*size)])
        return s[0], s[1], s[2]

# Get shared parameters, and split them to fit 3 gates, in the order of z, r, \tilde{h} (n stands for \tilde{h} in code)
def get_params(self, W, U):
    self.wz, self.wr, self.wn = self.split_params(W, self.hidden_size)
    self.uz, self.ur, self.un = self.split_params(U, self.input_size)
```

### 反向传播

```python
def backward(self, h_p, in_grad):
    self.dzz = in_grad * (self.n - h_p) * self.z * (1 - self.z)
    self.dzn = in_grad * self.z * (1 - self.n * self.n)
    self.dzr = np.dot(self.dzn, self.wn.T) * h_p * self.r * (1 - self.r)

    self.dwn = np.dot((self.r * h_p).T, self.dzn)
    self.dun = np.dot(self.x.T, self.dzn)
    self.dwr = np.dot(h_p.T, self.dzr)
    self.dur = np.dot(self.x.T, self.dzr)
    self.dwz = np.dot(h_p.T, self.dzz)
    self.duz = np.dot(self.x.T, self.dzz)

    self.merge_params()

    # pass to previous time step
    self.dh = in_grad * (1 - self.z) + np.dot(self.dzn, self.wn.T) * self.r + np.dot(self.dzr, self.wr.T) + np.dot(self.dzz, self.wz.T)
    # pass to previous layer
    self.dx = np.dot(self.dzn, self.un.T) + np.dot(self.dzr, self.ur.T) + np.dot(self.dzz, self.uz.T)
```

我们将所有拆分的参数merge到一起，便于更新梯度。

```python
def merge_params(self):
    self.dW = np.concatenate((self.dwz, self.dwr, self.dwn), axis=0)
    self.dU = np.concatenate((self.duz, self.dur, self.dun), axis=0)
```

## 最终结果

图20-8展示了训练过程，以及loss和accuracy的曲线变化。

![&#x56FE;20-8 loss&#x548C;accuracy&#x7684;&#x66F2;&#x7EBF;&#x53D8;&#x5316;&#x56FE;](.gitbook/assets/image%20%2813%29.png)

该模型在验证集上可得100%的正确率。网络正确性得到验证。

```python
  x1: [1, 1, 1, 0]
- x2: [1, 0, 0, 0]
------------------
true: [0, 1, 1, 0]
pred: [0, 1, 1, 0]
14 - 8 = 6
====================
  x1: [1, 1, 0, 0]
- x2: [0, 0, 0, 0]
------------------
true: [1, 1, 0, 0]
pred: [1, 1, 0, 0]
12 - 0 = 12
====================
  x1: [1, 0, 1, 0]
- x2: [0, 0, 0, 1]
------------------
true: [1, 0, 0, 1]
pred: [1, 0, 0, 1]
10 - 1 = 9
```

## keras实现

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt

from MiniFramework.DataReader_2_0 import *

from keras.models import Sequential
from keras.layers import GRU, Dense

train_file = "../data/ch19.train_minus.npz"
test_file = "../data/ch19.test_minus.npz"


def load_data():
    dataReader = DataReader_2_0(train_file, test_file)
    dataReader.ReadData()
    dataReader.Shuffle()
    dataReader.GenerateValidationSet(k=10)
    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev
    return x_train, y_train, x_test, y_test, x_val, y_val


def build_model():
    model = Sequential()
    model.add(GRU(input_shape=(4,2),
                        units=4))
    model.add(Dense(4, activation='sigmoid'))
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

#画出训练过程中训练和验证的精度与损失
def draw_train_history(history):
    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()


def test(x_test, y_test, model):
    print("testing...")
    count = x_test.shape[0]
    result = model.predict(x_test)
    r = np.random.randint(0, count, 10)
    for i in range(10):
        idx = r[i]
        x1 = x_test[idx, :, 0]
        x2 = x_test[idx, :, 1]
        print("  x1:", reverse(x1))
        print("- x2:", reverse(x2))
        print("------------------")
        print("true:", reverse(y_test[idx]))
        print("pred:", reverse(result[idx]))
        x1_dec = int("".join(map(str, reverse(x1))), 2)
        x2_dec = int("".join(map(str, reverse(x2))), 2)
        print("{0} - {1} = {2}".format(x1_dec, x2_dec, (x1_dec - x2_dec)))
        print("====================")
    # end for


def reverse(a):
    l = a.tolist()
    l.reverse()
    return l


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, x_val, y_val = load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(x_val.shape)

    model = build_model()
    history = model.fit(x_train, y_train,
                        epochs=200,
                        batch_size=64,
                        validation_data=(x_val, y_val))
    print(model.summary())
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    test(x_test, y_test, model)
```

### 模型输出

```python
test loss: 0.6068302603328929, test accuracy: 0.623161792755127
```

### 损失以及准确率曲线

![](.gitbook/assets/image%20%2855%29.png)

## 代码位置

原代码位置：[ch20, Level2](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch20-RNNModel/Level2_GRU_BinaryNumberMinus.py)

个人代码：[**GRU\_BinaryNumberMinus**](https://github.com/Knowledge-Precipitation-Tribe/Recurrent-neural-network/blob/master/code/GRU_BinaryNumberMinus.py)\*\*\*\*

