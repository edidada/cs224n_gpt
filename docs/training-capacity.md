# 全量训练资源测量

2026-11-01，GitHub 标准 Ubuntu runner，4 个逻辑 CPU，PyTorch 使用 2 线程，CPU 构建。仓库 self-hosted runner 列表为空。

| 任务 | 完整训练数据条数 | 10 epoch 纯训练粗估小时 |
| --- | ---: | ---: |
| SST | 8544 | 12.35 |
| CFIMDB | 1707 | 4.11 |
| Quora | 283003 | 426.68 |
| Sonnets | 131 | 0.33 |

测量方法：从最多 256 个均匀分布样本中，按 token 长度排序，取 10/30/50/70/90 分位各一步；使用真实 GPT-2、全参数反向传播及 AdamW 更新，batch size=1、最大上下文 1024。用五步平均时间乘以完整样本数和 10 epoch。估算不是训练成绩；不含评估、生成、下载、检查点 IO，且包含首步优化器初始化开销。冻结骨干训练不适用这张全参数耗时表。

证据：
- SST/CFIMDB：[成功测速运行](https://github.com/edidada/cs224n_gpt/actions/runs/41317631931)，artifact `capacity-sst`、`capacity-cfimdb`。
- Quora/Sonnets：[成功测速运行](https://github.com/edidada/cs224n_gpt/actions/runs/41317440575)，artifact `capacity-paraphrase`、`capacity-sonnet`。
- [全量 Part 1 训练](https://github.com/edidada/cs224n_gpt/actions/runs/41317349597)。
- [全量 Part 2 训练](https://github.com/edidada/cs224n_gpt/actions/runs/41317341050)。

GitHub 托管 runner 单 job 最长 6 小时，见 [官方限制](https://docs.github.com/en/enterprise-cloud%40latest/actions/reference/limits)。因此当前 CPU 配置无法在单个 job 内完成 Quora 10 epoch；不能以小样本成功替代全量实验。

## 指定 GPU runner

两个 lab yml 的 `workflow_dispatch` 提供：
- `full_training=true`：启动完整训练，普通 push 不重复启动全量任务。
- `runner_labels`：JSON 数组，例如 `["self-hosted","linux","x64","gpu"]`，需要账号已有这些 labels 对应的 runner。
- `accelerator=cuda121`：安装 PyTorch CUDA 12.1 构建并传入 `--use_gpu`；CPU 模式为 `cpu`。
- `epochs=10`：完整训练 epoch 数。

这是 runner 接入入口，未注册或验证 GPU runner，也未开通付费算力。全量训练完成与否必须以对应训练 job 的最终状态和完整日志/预测文件判断。
