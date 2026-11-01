# CS224N GPT-2 实验

上游课程项目分两部分；实验分支为 `edidada/part-1` 和 `edidada/part-2`，各有独立 workflow。

Part 1 手工实现因果多头注意力、pre-LN Transformer、位置嵌入、共享词表投影、AdamW 和情感分类头。验证包含上游优化器数值参考、微型 Hugging Face GPT-2 全位置数值比较、未来 token 隔离、padding 池化、反向梯度，以及真实 GPT-2 预训练权重 sanity check。

安装：Python 3.11，先安装适合平台的 PyTorch 2.5.1，再 `pip install -r requirements.txt`。
完整情感训练：`python classifier.py --fine-tune-mode last-linear-layer`，然后运行 full-model 配置。

CI 正确性测试不能代替 SST/CFIMDB 完整训练准确率、Quora 全量评估或诗歌质量人工评估。没有完整训练结果时，不宣称达到课程准确率或完成课程验收。

Part 2 实现二分类头与共享词表语言模型头；修复训练标签错误（yes/no 词表 ID 不能直接作为二分类类别）、训练/测试 prompt 不一致、padding 参与语言模型损失、生成结果丢失前三个字符等问题。新增可选 top-k 采样，与已有 temperature/top-p 可组合。模型检查点仅加载自己生成的可信文件。

## 完整实验命令

以下为完整训练，计算开销明显高于 CI 小样本验证；有 GPU 时添加 `--use_gpu`。建议先运行 `python optimizer_test.py`、`python -m pytest tests -q` 和 `python sanity_check.py`。

```bash
# Part 1：每条命令依次训练 SST 和 CFIMDB，并输出 dev/test 预测。
python classifier.py --fine-tune-mode last-linear-layer --lr 1e-3 --epochs 10
python classifier.py --fine-tune-mode full-model --lr 1e-5 --epochs 10
# Part 2：Quora 二分类；按 dev 最佳准确率保存检查点。
python paraphrase_detection.py --epochs 10 --lr 1e-5
# Part 2：sonnet 语言模型，默认 temperature/top-p，top-k=20 为采样扩展。
python sonnet_generation.py --epochs 10 --temperature 0.8 --top_p 0.9 --top_k 20
```

Part 1 应检查 `predictions/` 中两个训练模式、两个数据集各自的 dev/test 文件，共八个文件。Part 2 的输出为 `para-dev-output.csv`、`para-test-output.csv` 和 `generated_sonnets.txt`。不得把 test 标签用于调参。

## Actions 验证范围

- Part 1：上游优化器参考、微型模型数值/因果/反向传播测试、真实预训练 GPT-2 sanity check。
- Part 2：增加分类头梯度、训练/测试 prompt 一致性、整数标签、生成提示保留与上下文边界测试。
- `scripts/train_smoke.py`：真实 GPT-2 + 课程数据；SST、CFIMDB、Quora 各取 16 个训练和 16 个 dev 样本，冻结骨干训练分类头 20 步；一首训练诗截断 64 token，全模型更新 2 步。分类训练损失必须下降，诗歌损失必须有限。
- 同一个 held-out dev 提示、同一个种子生成 top-k=0/20 的 32-token 续写，供人工比较。只比较采样方法，不据此声称诗歌质量提高。
- artifact `real-data-smoke` 保存 JSON 指标和两份续写。小样本准确率不代表全量成绩，诗歌两步损失含 dropout，不要求单调下降。

完整课程报告还需要全量训练、dev 指标、消融比较和人工质量评价。当前 CI 用来确认实现与数据链路，不能当作这些工作的替代结果。
