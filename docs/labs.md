# CS224N GPT-2 实验

上游课程项目分两部分；实验分支为 `edidada/part-1` 和 `edidada/part-2`，各有独立 workflow。

Part 1 手工实现因果多头注意力、pre-LN Transformer、位置嵌入、共享词表投影、AdamW 和情感分类头。验证包含上游优化器数值参考、微型 Hugging Face GPT-2 全位置数值比较、未来 token 隔离、padding 池化、反向梯度，以及真实 GPT-2 预训练权重 sanity check。

安装：Python 3.11，先安装适合平台的 PyTorch 2.5.1，再 `pip install -r requirements.txt`。
完整情感训练：`python classifier.py --fine-tune-mode last-linear-layer`（以脚本 --help 的实际参数名为准），然后运行 full-model 配置。

CI 正确性测试不能代替 SST/CFIMDB 完整训练准确率、Quora 全量评估或诗歌质量人工评估。没有完整训练结果时，不宣称达到课程准确率或完成课程验收。
