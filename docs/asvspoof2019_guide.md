# ASVspoof2019 训练与推理操作手册

本文面向使用 Tomato 框架在 ASVspoof2019 数据集上完成训练与推理的流程需求，覆盖环境准备、数据整理、配置说明、训练与推理命令以及常见问题排查。

## 1. 环境准备

1. 建议使用 Python ≥ 3.9：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. 安装依赖：
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. 若需要 GPU 训练，请确认 CUDA/cuDNN 与 PyTorch 版本兼容。

## 2. 数据准备

1. 获取官方 ASVspoof2019 数据集（Logical Access, LA），参考官方发布页面下载 wav 与协议文件。
2. 将音频按官方目录结构整理，例如：
   ```
   dataset_root/
     ASVspoof2019_LA_train/flac/
     ASVspoof2019_LA_dev/flac/
     ASVspoof2019_LA_eval/flac/
   ```
3. 准备元信息：Tomato 默认读取 `dataset_meta/ASVspoof2019` 下的 `{train,dev,test}.tsv` 与 `{train,dev,test}.txt`。
   - 若已有官方协议，可使用自定义脚本转换为上述格式。
   - 示例模板可参考 `example/example_data`。
4. 将 `dataset_meta` 目录放置在仓库根目录或在配置文件中写绝对路径。

## 3. 配置说明

基础配置文件位于 `example/example.yaml`，关键字段如下：

- `data.data_path`：指向包含 TSV/TXT 的目录，例如 `dataset_meta/ASVspoof2019`。
- `data.augment_type`、`data.do_augment`：控制是否启用 RawBoost 等增强。
- `model.frontend_path`：若使用预训练前端（如 XLS-R 300M），需要提供权重路径。
- `train` 段落中的学习率、最大 epoch、日志间隔等可根据资源调整。
- 如需多卡训练，可在运行命令时设置 `CUDA_VISIBLE_DEVICES` 并调整脚本逻辑。

建议复制 `example/example.yaml` 为自定义文件，例如 `configs/asvspoof2019_xlsr.yaml`，避免修改示例配置。

## 4. 训练流程

1. 确保前述配置与数据路径正确。
2. 执行训练命令（以 OC 任务为例）：
   ```bash
   python train.py \
     -c configs/asvspoof2019_xlsr.yaml \
     -task oc \
     -exp asvspoof2019_xlsr_oc \
     -s ckpts
   ```
   - `-exp` 用于指定实验名，输出位于 `output/ckpts/<exp>/seed<seed>`。
   - `-task` 支持 `oc`（One-Class）与 `xent`（交叉熵）。
   - 若需恢复训练，可追加 `-ckpt output/ckpts/<exp>/seed<seed>`。
3. 训练过程中会自动：
   - 写入 `train.log`、TensorBoard 日志。
   - 保存 `best.mdl`、`last.mdl` 等模型文件。
   - 生成层间置信度可视化（若模型支持）。

## 5. 推理流程

1. 准备训练阶段保存的 checkpoint 目录（包含 `best.mdl` 等文件）。
2. 运行推理命令（以 `best` checkpoint、test split 为例）：
   ```bash
   python infer.py \
     -task oc \
     -c configs/asvspoof2019_xlsr.yaml \
     -exp infer_asvspoof2019 \
     -ckpt output/ckpts/asvspoof2019_xlsr_oc \
     -ckpt_tag best \
     -o output/infer/asvspoof2019_xlsr_oc \
     -tag test_run \
     -s test
   ```
   - 结果 CSV 将保存到 `output/infer/<exp>/test_run.csv`。
   - 输出包含 `prediction` 与 `label`，可用于后续评估。

## 6. 评估建议

- 使用仓库内 `tomato/train_util/` 中的工具计算 EER、AUC 等指标，或外部脚本解析推理 CSV。
- 若需对不同子集（攻击类型、说话人）评估，可根据 CSV 中的 uttid 与额外元信息进行聚合。
- 推荐将 TensorBoard 日志与层间置信度热力图结合，观察模型在训练过程中的学习重点变化。

## 7. 常见问题

| 问题 | 排查要点 |
| --- | --- |
| 无法找到 TSV/TXT | 检查 `data.data_path` 是否指向包含元信息的目录，文件名大小写需一致 |
| 加载前端失败 | 确认 `model.frontend_path` 存在且与配置匹配（例如 XLS-R 300M） |
| 训练卡死或显存不足 | 减小 `data.batch_size` 或 `data.max_samples`，必要时关闭增强 |
| 推理输出为空 | 确保 `-s` 对应的数据 split 在配置的数据加载逻辑中存在 |
| 指标异常 | 检查标签是否与任务定义一致（Fake=1/Real=0），并验证 CSV 解析脚本 |

## 8. 自动化建议

- 通过 Shell 脚本一次性启动多次实验：
  ```bash
  for seed in 0 1 2; do
    python train.py -c configs/asvspoof2019_xlsr.yaml -task oc -exp asvspoof2019_seed${seed} -s ckpts
  done
  ```
- 结合 `tools/layer_confidence_heatmap.py` 对多个实验的层间置信度曲线进行比较，辅助模型选择。

按以上步骤，即可复现 ASVspoof2019 数据集上的训练与推理流程并执行可靠的性能评估。
