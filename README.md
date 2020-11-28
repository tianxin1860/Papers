 # Papers

 ### Representation Learning

| Paper | 核心思想 | 备注 |
| --- | --- |--- |
|[CoVe](https://einstein.ai/static/images/pages/research/cove/McCann2017LearnedIT.pdf)  | 基于2层双向 LSTM 预训练翻译模型作为 embedding encoder | 2017-NIPS|
|[ELMo](https://arxiv.org/pdf/1802.05365.pdf)  | 基于2层双向 LSTM 预训练 Langeage Model 作为 embedding encoder | 2018-NAACL Best Paper|
|[GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)  | 基于12层 Transformer Decoder 预训练 Langeage Model 作为 embedding encoder | 2018-OpenAI |
|[BERT](https://arxiv.org/pdf/1810.04805.pdf)  |  基于双向 Transformer 预训练 Masked Language Model 作为 embedding encoder |2019-NAACL Best Paper|
|[Span-BERT](https://arxiv.org/pdf/1907.10529.pdf) |核心思想是随机对文本片段进行 Mask, 而不是对 subword/word 进行 Mask, 与 ERENI 提出的思想没有太大区别; 文中提出的 Span Boundary Objective 效果并不是特别明显，略显牵强; 去掉 NSP 任务后显著提升 QA 任务效果是因为单句长度扩展到了 512，获得了更多的上下文信息|2019-arXiv Facebook AI Research|
|[MT-DNN](https://arxiv.org/pdf/1901.11504.pdf)  |  基于 BERT 利用 multi-task finetune 提升 embedding 的领域泛化性 |2019-arXiv|
|[Transformer-XL](https://arxiv.org/abs/1901.02860)  | 通过引入 segment-level recurrence 机制解决了标准 Transformer 最大长度受限的问题(文章强调evaluation 阶段速度比标准 transformer 快 1800 倍, 同时解决了在计算 attention score 的时候，如何融入 relative position 信息的问题)|2019-ACL|
|[Universal Transformer](https://arxiv.org/abs/1807.03819)  | Todo| 2019-ICLR|
|[GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Todo | 2019-OpenAI|
|[XLM](https://arxiv.org/abs/1901.07291)| 通过引入平行语料训练 Translation LM, 使用多语数据训练 CLM、MLM、TLM 提升 XNLI 任务效果| 2019-Facebook AI Research |
|[MASS](https://arxiv.org/abs/1905.02450)| 在 encoder 端 Mask 句中连续片段, 在 decoder 端只输入在 encoder 端被 mask 掉的 token 来训练生成模型, 从而让 BERT 可以用于生成任务| 2019-ICML |
|[UNILM](https://arxiv.org/abs/1905.03197)| 通过 Mask LM、Uni-directional LM、SeqSeq 联合训练, 来构建统一的语言模型, 同时适合 NLU 任务和 NLG 任务| 2019-arXiv |
|[UNILM-v2](https://arxiv.org/abs/2002.12804) | 通过 1 次 forward 操作同时训练基于 Mask-Span 的 MLM 和 Uni-directional LM 任务, Block-wise Permutation 策略本质是一种数据增强 | 2020-arXiv |
|[Info-XLM](https://arxiv.org/pdf/2007.07834.pdf) | 从最大化互信息的视角统一了 MMLM 、TLM 任务的训练目标，同时新提出基于平行语料 Sentence-Level 的互信息建模任务 | 2020-arXiv |
|[Adaptive Attention Span in Transformers](https://arxiv.org/abs/1905.07799)| Todo | 2019-arXiv |
|[XLNet](https://arxiv.org/pdf/1906.08237.pdf)| 通过构造适当的 Attention Mask 矩阵来模拟 Permutation LM 任务, 兼顾了 AE 模型(可以获得上下文信息)和 AR 模型(适合 NLG 任务) 的优点| 2019-arXiv |
|[Evaluation of sentence embeddings in downstream and linguistic probing tasks](https://arxiv.org/pdf/1806.06259.pdf)| 在[SentEval](https://github.com/facebookresearch/SentEval) 5 大类公开数据集上评估了 Word2Vec、Glove、FastText、p-mean、SkipThought、InferSent、ELMo、USE 这些模型产出的 sentence embedding 效果，在英文数据上对 sentence embedding 给出了一个较为扎实的基线, 技术选型时可作为参考 | 2018-arXiv |
|[Universal Sentence Encoder](https://arxiv.org/pdf/1803.11175.pdf)| 基于 Transformer encoder 为网络基础，采用类似 skip-thought 这样的自监督任务、以及 SNLI 等监督任务为训练目标，产出通用的 Sentence Encoder; 核心论点: 通用性体现在 multi-task pretrain 上(这个pretrain 全是句子级别的任务，没有 token 级别的任务)| 2018-arXiv |


### Optimizaiton
### Loss
| Paper | 核心思想 | 备注 |
| --- | --- |--- |
|[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)| 通过预测概率自适应调整 Loss 大小, 使得容易预测对的样本 loss 很小，可以动态解决样本不均衡问题和样本学习难易程度不同的问题，区分出随机负样本和强负样本对模型的 Loss 贡献 | 2017-ICCV |

### Performance Optimization about Deep Learning
| Paper | 核心思想 | 备注 |
| --- | --- |--- |
|[Mixed Precision Training](https://arxiv.org/abs/1710.03740)|深度学习的计算过程对数值精度的要求并不高，使用 float 16 代替 float 32 进行计算，可以利用充分利用 TensorCore 硬件支持实现巨大加速，同时显著降低显存占用| 2018-ICLR |
|[Ring All-Reduce](http://on-demand.gputechconf.com/gtc/2017/presentation/s7543-andrew-gibiansky-effectively-scakukbg-deep-learning-frameworks.pdf)| 基于 Ring All-Reduce 算法实现多 GPU 通信量不随着 GPU 卡数的增加而增加，同时避免 Pserver 模式中心化带来的通信瓶颈 | Tutorial|
|[Hierarchical All-Reduce](http://learningsys.org/nips18/assets/papers/6CameraReadySubmissionlearnsys2018_blc.pdf)| Todo | 2018-ICLR |
|[Tree All-Reduce](https://web.ece.ucdavis.edu/~ctcyang/pub/amaz-techreport2018.pdf)| Todo | 2018-AmozonTechReport |
|[TICTAC](https://www.sysml.cc/doc/2019/199.pdf)| Todo | 2019-SysML |
|[Quantized Neural Networks:Training Neural Networks with Low Precision Weights and Activations](https://arxiv.org/pdf/1609.07061.pdf)| Todo | 2016-arXiv |


### Matching
| Paper | 核心思想 | 备注 |
| --- | --- |--- |
|[DOMAIN-RELEVANT EMBEDDINGS FOR QUESTION SIMILARITY](https://openreview.net/pdf?id=Byxn9CNYPr)|如何在仅有少量监督数据的特定领域训练出好的 Similarity Model？基于 BERT 进行 Dobule finetune, First Fitune 时构造一些与 Target Domain  相关的 semi-supervised 任务或者 supervised 任务, Second Fintune 时在 target task 上训练即可 | 2020-ICLR Under review|

### DISTILLING
| Paper | 核心思想 | 备注 |
| --- | --- |--- |
|[TINYBERT](https://arxiv.org/pdf/1909.10351.pdf)|| 2019-Arxiv Huawei Noah’s Ark Lab 刘群|
|[ALBERT](https://arxiv.org/pdf/1909.11942.pdf)|| 2019-Arxiv Google Zhenzhong Lan|

### Quantization
| Paper | 核心思想 | 备注 |
| --- | --- |--- |
|[Making Neural Nets Work With Low Precision](https://sahnimanas.github.io/post/quantization-in-tflite/)|||
|[Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/pdf/1712.05877.pdf)|||
|[TRAINING AND INFERENCE WITH INTEGERS IN DEEP NEURAL NETWORKS](https://arxiv.org/pdf/1802.04680.pdf)||2018-ICLR 清华|

 ### Infomation Retrieval
| Paper | 核心思想 | 备注 |
| --- | --- |--- |
|[APPROXIMATE NEAREST NEIGHBOR NEGATIVE CONTRASTIVE LEARNING FOR DENSE TEXT RETRIEVAL](https://openreview.net/pdf?id=zeFrfgyZln)|语义索引的核心问题在于如何保证训练阶段与应用阶段的负样本分布尽可能接近? 本文通过对 docs 的索引进行高频更新，训练过程中不断挖掘出更 Informative 的负样本解决语义索引的核心问题 | 2021-ICLR Microsoft|

 ### Deep Learning System
#### Course
| 课程名 | 备注 |
| --- | --- |
|[cs294 AI-Sys Spring 2019](https://ucbrise.github.io/cs294-ai-sys-sp19/)  | |
|[CSE 599W: Systems for ML](http://dlsys.cs.washington.edu/schedule)  | |


#### Paper
| Paper | 备注 |
| --- | --- |
|[TensorFlow: A System for Large-Scale Machine Learning](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)  | TensorFlow 白皮书 |
|[TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](http://download.tensorflow.org/paper/whitepaper2015.pdf)  | TensorFlow 白皮书 |
|[MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems](https://www.cs.cmu.edu/~muli/file/mxnet-learning-sys.pdf)  | MXNet |
|[On-the-fly Operation Batching in Dynamic Computation Graphs](https://papers.nips.cc/paper/6986-on-the-fly-operation-batching-in-dynamic-computation-graphs.pdf)  |  |
|[TENSORFLOW EAGER: A MULTI-STAGE, PYTHON-EMBEDDED DSL FOR MACHINE LEARNING](https://www.sysml.cc/doc/2019/88.pdf)  | 2019-SysML |
|[AUTOGRAPH: IMPERATIVE-STYLE CODING WITH GRAPH-BASED PERFORMANCE](https://www.sysml.cc/doc/2019/194.pdf)  | 2019-SysML |
|[PYTORCH-BIGGRAPH: A LARGE-SCALE GRAPH EMBEDDING SYSTEM](https://www.sysml.cc/doc/2019/71.pdf)  | 2019-SysML |
