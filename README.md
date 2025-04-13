# 三层神经网络图像分类器 (CIFAR-10)

本项目使用 NumPy 手动实现三层神经网络进行图像分类任务。

## 📁 项目结构

```
.
├── figure        	# 图片
├── model.py        # 神经网络模型定义
├── train.py        # 训练主程序
├── test.py         # 测试主程序
├── utils.py        # 数据加载与工具函数
├── README.md       # 使用说明
```

## 📦 环境依赖

- Python >= 3.7
- NumPy >= 1.20
- Matplotlib >= 3.0（用于训练过程可视化）

安装方式（可选）：

```bash
pip install numpy matplotlib
```

## 📥 数据准备

从官网获取 [CIFAR-10 数据集](https://www.cs.toronto.edu/~kriz/cifar.html)，并解压至本目录下的 `cifar-10-batches-py/` 文件夹。

## 🏃‍♂️ 训练模型

#### 1.训练一个自定义模型：

```bash
python train.py --lr 0.1 --hidden_size 256 --reg 0.0 --activation relu --epochs 30 --batch_size 128 --verbose --plot
```

##### 支持的参数说明：

| 参数名          | 含义                               | 示例                    |
| --------------- | ---------------------------------- | ----------------------- |
| `--lr`          | 学习率                             | `0.001`,`0.01`, `0.1`   |
| `--epochs`      | 训练轮数                           | `30`, `50`              |
| `--batch_size`  | 每轮训练的批大小                   | `64`, `128`             |
| `--hidden_size` | 隐藏层大小                         | `64`, `128`,`256`       |
| `--reg`         | L2 正则化强度                      | `1e-3`, `1e-2`,`0`      |
| `--activation`  | 激活函数，支持 `relu/sigmoid/tanh` | `relu`,`sigmoid`,`tanh` |

训练完成后，最优模型将保存为 best_model.npz，并生成训练过程中的Train Loss curve和Validation Accuracy curve。

#### 2.启用网格搜索（忽略其它参数）：

```bash
python train.py --grid
```

## 🧪 测试模型

```bash
python test.py
```

将输出best model在测试集上的准确率，并生成模型参数的可视化图像。

