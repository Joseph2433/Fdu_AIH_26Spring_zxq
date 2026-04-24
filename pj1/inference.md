# Inference 使用说明

本文档简要说明如何使用现有测试脚本进行评估。

## 1. 测试集目录要求

测试集结构需与训练集一致，即：

- test/1/*.bmp
- test/2/*.bmp
- ...
- test/12/*.bmp

脚本会使用模型中保存的 label_map 对类别进行对齐。

## 2. MLP 测试（part1_2）

文件：p1/part1_2_test.py

在 main() 中修改：

- model_path = "part1_2_model.pkl"
- test_dir = "../test"

然后在 p1 目录运行：

```bash
python .\part1_2_test.py
```

输出内容：

- Test Acc
- Test Loss
- Per-class accuracy

## 3. CNN 测试（part2）

文件：p2/part2_cnn_test.py

在 main() 中修改：

- model_path = "part2_cnn.pt"
- test_dir = "../test"
- batch_size = 128
- device_name = "auto"（可改为 "cpu" 或 "cuda"）

然后在 p2 目录运行：

```bash
python .\part2_cnn_test.py
```

输出内容：

- Test Loss
- Test Acc
- Per-class accuracy

## 4. 常见问题

1. 提示“测试目录不存在”
- 检查 test_dir 是否为相对当前运行目录的正确路径。

2. 提示“未读取到任何带标签样本”
- 检查目录层级是否为 test/类别名/*.bmp。

3. 导入库未解析
- 通常是当前 Python 环境未安装依赖或解释器未切换到正确环境。
