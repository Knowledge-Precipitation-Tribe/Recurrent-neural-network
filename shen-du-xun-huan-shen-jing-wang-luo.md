# 深度循环神经网络

## 深度循环神经网络的结构图

前面的几个例子中，单独看每一时刻的网络结构，其实都是由“输入层-&gt;隐层-&gt;输出层”所组成的，这与我们在前馈神经网络中学到的单隐层的知识一样，由于输入层不算做网络的一层，输出层是必须具备的，所以网络只有一个隐层。我们知道单隐层的能力是有限的，所以人们会使用更深（更多隐层）的网络来解决复杂的问题。

在循环神经网络中，会有同样的需求，要求每一时刻的网络是由多个隐层组成。比如图19-20为两个隐层的循环神经网络，用于解决空气质量预测问题。

![&#x56FE;19-20 &#x4E24;&#x4E2A;&#x9690;&#x5C42;&#x7684;&#x5FAA;&#x73AF;&#x795E;&#x7ECF;&#x7F51;&#x7EDC;](.gitbook/assets/image%20%2827%29.png)

注意图19-20中最左侧的两个隐藏状态s1和s2是同时展开为右侧的图的， 这样的循环神经网络称为深度循环神经网络，它可以具备比单隐层的循环神经网络更强大的能力。

## 前向计算

### 公式推导

对于第一个时间步： 

$$ h1 = x \cdot U \tag{1} $$

$$ h2 = s1 \cdot Q \tag{2} $$

对于后面的时间步： 

$$ h1 = x \cdot U + s1_{t-1} \cdot W1 \tag{3} $$

$$ h2 = s1 \cdot Q + s2_{t-1} \cdot W2 \tag{4} $$

对于所有的时间步：

$$ s1 = Tanh(h1) \tag{5} $$

$$ s2 = Tanh(h2) \tag{6} $$

对于最后一个时间步： 

$$ z = s2 \cdot V \tag{7} $$

$$ a = Identity(z) \tag{8} $$

$$ Loss = loss_{\tau} = \frac{1}{2} (a-y)^2 \tag{9} $$

由于是拟合任务，所以公式8的Identity\(\)函数只是简单地令a=z，以便统一编程接口，最后用均方差做为损失函数。

注意并不是所有的循环神经网络都只在最后一个时间步有监督学习信号，而只是我们这个问题需要这样。在19.2节中的例子就是需要在每一个时间步都要有输出并计算损失函数值的。所以，公式9中只计算了最后一个时间步的损失函数值，做为整体的损失函数值。

### 代码实现

注意前向计算时需要把prev\_s1和prev\_s2传入，即上一个时间步的两个隐层的节点值（矩阵）。

```python
class timestep(object):
    def forward(self, x, U, V, Q, W1, W2, prev_s1, prev_s2, isFirst, isLast):
        ...
```

## 反向传播

### 公式推导

反向传播部分和前面章节的内容大致相似，我们只把几个关键步骤直接列出来，不做具体推导：

对于最后一个时间步： 

$$ \frac{\partial Loss}{\partial z} = a-y \rightarrow dz \tag{10} $$

$$ \frac{\partial Loss}{\partial V}=\frac{\partial Loss}{\partial z}\frac{\partial z}{\partial V}=s2^T \cdot dz \rightarrow dV \tag{11} $$

$$ \begin{aligned} \frac{\partial Loss}{\partial h2} = \frac{\partial Loss}{\partial z}\frac{\partial z}{\partial s2}\frac{\partial s2}{\partial h2} \ &=(dz \cdot V^T) \odot \sigma'(s2) \rightarrow dh2 \end{aligned} \tag{12} $$

$$ \begin{aligned} \frac{\partial Loss}{\partial h1} = \frac{\partial Loss}{\partial h2}\frac{\partial h2}{\partial s1}\frac{\partial s1}{\partial h1} \ &=(dh2 \cdot Q^T) \odot \sigma'(s1) \rightarrow dh1 \end{aligned} \tag{13} $$

对于其他时间步：

$$ dz = 0 \tag{14} $$

$$ \begin{aligned} \frac{\partial Loss}{\partial h2_t} = \frac{\partial Loss}{\partial h2_{t+1}}\frac{\partial h2_{t+1}}{\partial s2_t}\frac{\partial s2_t}{\partial h2_t} \ &=(dh2_{t+1} \cdot W2^T) \odot \sigma'(s2_t) \rightarrow dh2_t \end{aligned} \tag{15} $$

$$ dV = 0 \tag{16} $$

$$ \begin{aligned} \frac{\partial Loss}{\partial h1_t} = \frac{\partial Loss}{\partial h1_{t+1}}\frac{\partial h1_{t+1}}{\partial s1_t}\frac{\partial s1_t}{\partial h1_t} \ &=(dh1_{t+1} \cdot W1^T) \odot \sigma'(s1_t) \rightarrow dh1_t \end{aligned} \tag{17} $$

对于第一个时间步：

$$ dW1 = 0, dW2 = 0 \tag{18} $$

对于其他时间步：

$$ \frac{\partial Loss}{\partial W1}=s1^T_{t-1} \cdot dh_1 \rightarrow dW1 \tag{19} $$

$$ \frac{\partial Loss}{\partial W2}=s2^T_{t-1} \cdot dh2 \rightarrow dW2 \tag{20} $$

对于所有时间步：

$$ \frac{\partial Loss}{\partial Q}=\frac{\partial Loss}{\partial h2}\frac{\partial h2}{\partial Q}=s1^T \cdot dh2 \rightarrow dQ \tag{21} $$

$$ \frac{\partial Loss}{\partial U}=\frac{\partial Loss}{\partial h1}\frac{\partial h1}{\partial U}=x^T \cdot dh1 \rightarrow dU \tag{22} $$

### 代码实现

```python
class timestep(object):
    def backward(self, y, prev_s1, prev_s2, next_dh1, next_dh2, isFirst, isLast):
        ...
```

## 运行结果

### 超参设置

我们搭建一个双隐层的循环神经网络，隐层1的神经元数为2，隐层2的神经元数也为2，其它参数保持与单隐层的循环神经网络一致：

* 网络类型：回归
* 时间步数：24
* 学习率：0.05
* 最大迭代数：100
* 批大小：64
* 输入特征数：6
* 输出维度：1

### 训练结果

训练过程如图19-21所示，训练结果如表19-10所示。

![&#x56FE;19-21 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x7684;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x5EA6;&#x7684;&#x53D8;&#x5316;](.gitbook/assets/image%20%2844%29.png)

表19-10 预测时长与准确度的关系

| 预测时长 | 结果 | 预测结果 |
| :--- | :--- | :--- |
| 8 | 损失函数值： 0.001157 准确度： 0.740684 | ![](.gitbook/assets/image%20%2829%29.png)  |
| 4 | 损失函数值： 0.000644 准确度： 0.855700 | ![](.gitbook/assets/image%20%2830%29.png)  |
| 2 | 损失函数值： 0.000377 准确度： 0.915486 | ![](.gitbook/assets/image%20%2818%29.png)  |
| 1 | 损失函数值： 0.000239 准确度： 0.946411 | ![](.gitbook/assets/image%20%2817%29.png)  |

### 与单层循环神经网络的比较

对于单层循环神经网络，参数配置如下：

```text
U: 6x4+4=28
V: 4x1+1= 5
W: 4x4  =16
-----------
Total:   49
```

对于两层的循环神经网络来说，参数配置如下：

```text
U: 6x2=12
Q: 2x2= 4
V: 2x1= 2
W1:2x2= 4
W2:2x2= 4
---------
Total: 26
```

表19-11 预测结果比较

|  | 单隐层循环神经网络 | 深度（双层）循环神经网络 |
| :--- | :--- | :--- |
| 参数个数 | 49 | 26 |
| 损失函数值（8小时） | 0.001171 | 0.001157 |
| 损失函数值（4小时） | 0.000686 | 0.000644 |
| 损失函数值（2小时） | 0.000414 | 0.000377 |
| 损失函数值（1小时） | 0.000268 | 0.000239 |
| 准确率值（8小时） | 0.737769 | 0.740684 |
| 准确率值（4小时） | 0.846447 | 0.855700 |
| 准确率值（2小时） | 0.907291 | 0.915486 |
| 准确率值（1小时） | 0.940090 | 0.946411 |

从表19-11可以看到，双层的循环神经网络在参数少的情况下，取得了比单层循环神经网络好的效果。

## keras实现

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt

from ExtendedDataReader.PM25DataReader import *

from keras.models import Sequential, load_model
from keras.layers import SimpleRNN

train_file = "../data/ch19.train_echo.npz"
test_file = "../data/ch19.test_echo.npz"


def load_data(net_type, num_step):
    dataReader = PM25DataReader(net_type, num_step)
    dataReader.ReadData()
    dataReader.Normalize()
    dataReader.GenerateValidationSet(k=1000)
    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev
    return x_train, y_train, x_test, y_test, x_val, y_val


def build_model():
    model = Sequential()
    model.add(SimpleRNN(input_shape=(24,6),
                        units=6,
                        return_sequences=True))
    model.add(SimpleRNN(units=1))
    model.compile(optimizer='Adam',
                  loss='mean_squared_error')
    return model

#画出训练过程中训练和验证的精度与损失
def draw_train_history(history):
    plt.figure(1)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()


def show_result(model, X, Y, num_step, pred_step, start, end):
    assert(X.shape[0] == Y.shape[0])
    count = X.shape[0] - X.shape[0] % pred_step
    A = np.zeros((count,1))

    for i in range(0, count, pred_step):
        A[i:i+pred_step] = predict(model, X[i:i+pred_step], num_step, pred_step)

    plt.plot(A[start+1:end+1], 'r-x', label="Pred")
    plt.plot(Y[start:end], 'b-o', label="True")
    plt.legend()
    plt.show()

def predict(net, X, num_step, pred_step):
    A = np.zeros((pred_step, 1))
    for i in range(pred_step):
        x = set_predicated_value(X[i:i+1], A, num_step, i)
        a = net.predict(x)
        A[i,0] = a
    #endfor
    return A

def set_predicated_value(X, A, num_step, predicated_step):
    x = X.copy()
    for i in range(predicated_step):
        x[0, num_step - predicated_step + i, 0] = A[i]
    #endfor
    return x


if __name__ == '__main__':
    net_type = NetType.Fitting
    num_step = 24
    x_train, y_train, x_test, y_test, x_val, y_val = load_data(net_type, num_step)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    # print(x_val.shape)
    # print(y_train.shape)

    model = build_model()
    history = model.fit(x_train, y_train,
                        epochs=5,
                        batch_size=64,
                        validation_data=(x_val, y_val))
    # model = load_model("pm25.h5")
    print(model.summary())
    model.save("deepRnn_pm25.h5")
    draw_train_history(history)

    loss = model.evaluate(x_test, y_test)
    print("test loss: {}".format(loss))

    pred_steps = [8,4,2,1]
    for i in range(4):
        show_result(model, x_test, y_test, num_step, pred_steps[i], 1050, 1150)
```

### 模型输出

```python
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
simple_rnn_1 (SimpleRNN)     (None, 24, 6)             78        
_________________________________________________________________
simple_rnn_2 (SimpleRNN)     (None, 1)                 8         
=================================================================
Total params: 86
Trainable params: 86
Non-trainable params: 0
_________________________________________________________________

test loss: 0.0011219596998781416
```

### 损失曲线

![](.gitbook/assets/image%20%2853%29.png)

## 代码位置

原代码位置：[ch19, Level6](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch19-RNNBasic/Level6_DeepRnn_PM25.py)

个人代码：[**DeepRnn\_PM25**](https://github.com/Knowledge-Precipitation-Tribe/Recurrent-neural-network/blob/master/code/DeepRnn_PM25.py)\*\*\*\*

