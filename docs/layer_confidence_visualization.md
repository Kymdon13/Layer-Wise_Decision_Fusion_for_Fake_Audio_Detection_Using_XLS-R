# 层间置信度热力图复现流程

本说明提供一个从环境搭建、数据准备、训练到生成层间置信度热力图的完整流程，帮助你可重复地获得可解释性结果。示例基于 `XLSRAdapter` 等带有可学习层权重（`gamma`）的模型，其他模型若实现了 `get_layer_confidence_snapshot()` 也可直接复用。

## 1. 环境搭建

1. **准备 Python 环境**（建议 Python ≥ 3.9）
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **安装依赖**（包含新增的 `matplotlib` 用于作图）
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 2. 数据准备

1. **下载/整理数据元信息**：默认配置假设已有 `dataset_meta/ASVspoof2019/{train,dev,test}.{tsv,txt}` 等文件。你可以使用仓库中的 `example/example_data` 作为最小可运行示例，只需在配置文件中调整为对应路径。
2. **下载前端模型权重**：若使用 `XLSRAdapter`，需要准备 XLS-R 300M 权重并在配置 `model.frontend_path` 中指明；可参考 `README.md` 提供的 HuggingFace 链接。

## 3. 训练示例

以下命令基于 `example/example.yaml`，请确保其中的数据路径、前端模型路径已按你的实际环境修改。

```bash
python train.py \
  -c example/example.yaml \
  -task oc \
  -exp demo_xlsr \
  -s ckpts
```

- 训练过程中将自动在 `output/ckpts/demo_xlsr/seed<seed>/` 目录保存模型、日志以及新增的层间置信度结果。
- 目前 `LayerConfidenceTracker` 会在每个 epoch 结束后记录一次 `gamma` 的 softmax，并在任务收尾时输出：
  - `layer_confidence.csv`：每行对应一个 epoch，列为 `epoch, layer_0, layer_1, ...`
  - `layer_confidence_heatmap.png`：以 epoch 为纵轴、层索引为横轴的热力图，颜色表示置信度大小。

## 4. 独立绘制热力图

如果需要在训练结束后重新绘制或自定义输出路径，可调用新增脚本：

```bash
python tools/layer_confidence_heatmap.py \
  output/ckpts/demo_xlsr/seed<seed>/layer_confidence.csv \
  --output output/ckpts/demo_xlsr/seed<seed>/layer_confidence_custom.png
```

脚本会读取 CSV 并生成新的 PNG 文件，默认文件名为 `<csv_stem>_heatmap.png`。

## 5. 结果解读建议

- **层选择趋势**：热力图列表示层索引，行表示 epoch，可快速观察模型在训练过程中如何调整不同层的权重。
- **收敛监控**：若某些层在早期权重波动较大但逐步稳定，说明模型在对具体层进行偏好选择。
- **模型比较**：不同实验（如不同前端冻结策略）生成的热力图可直接对比，辅助解释性能差异。

## 6. 常见问题

- **无热力图文件**：确认所用模型实现了 `get_layer_confidence_snapshot()`（`XLSRAdapter` 与 `XLSRTimeFirst` 已内置）。若返回值为 `None`，追踪器将跳过记录。
- **CSV 为空**：检查训练是否至少完成一个 epoch；若提前结束在第 0 个 epoch 中断，不会生成有效记录。
- **多阶段训练**：对于持续学习任务，我们在每个阶段结束后都会覆盖输出，确保热力图始终反映最新的完整训练轨迹。

按以上流程操作，即可稳定复现并可视化层间置信度，为模型做进一步的可解释性分析提供依据。
