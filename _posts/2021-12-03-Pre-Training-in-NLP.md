---
layout: post
title: Pre-Training in NLP
date: 2021-12-03 15:39
author: LiHan
categories: MachineLearning
---

预训练的概念受到人类生物的启发，由于先天的能力，我们不需要从头开始学习一切。相反，我们转移和重用我们过去学到的旧知识来理解新知识，处理各种新任务。

在人工智能中，预训练模仿了人类处理新知识的方式。即：使用以前学过的任务的模型参数来初始化新任务的模型参数。通过这种方式，旧的知识帮助新的模型，从旧的经验而不是从头开始执行新的任务。

预训练将模型学习分为两部分：

- 半监督学习（共性学习）：使用未标注数据对模型进行训练
- 监督学习（特性学习）：使用标注数据对上一步得到的模型进行微调

也许现在这两句话有些难以理解，没关系，我们将在接下来逐渐详细展开这一概念。

## Word2vec

> 该部分内容借鉴自 [Chris McCormick. Word2Vec Tutorial-The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)[^1]以及 [ Lilian Weng. Learning Word Embedding](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html)[^3]

Word2vec由Mikolov et al.于2013提出[^2]，使用了一种巧妙的思路对单词进行编码：我们的模型拥有隐藏层和输出层，并用一个“虚假任务”对模型进行训练，在训练完成之后将输出层“掀开”，隐藏层的输出就是我们期望的单词编码。

编码的目的是将单词映射到$N$维空间中，编码后的结果应能够很好的反应单词本身的含义，如相同含义的单词在空间内彼此接近，他们在空间中的距离应该能够与单词本身的相似程度相对应。

Word2vec可分为：`Skip-Gram Model`与`Continuous Bag-of-Words (CBOW)`两种，前者使用target word预测context，后者使用context预测target word，接下来仅讨论`Skip-Gram Model`。

**Fake  Task**

在Skip-Gram Model中，虚假任务定义为：针对某个单词，输出其他单词是其上下文的概率。

其中上下文使用滑动窗口提取，举个例子：

- “The quick brown fox jumps over the lazy dog.”

| Sliding window(size = 5)                           | Target word | Context                   | Training Samples(Skip-Gram) |
| -------------------------------------------------- | ----------- | ------------------------- | --------------------------- |
| [**The** quick brown] fox jumps over the lazy dog. | the         | quick, brown              | (the, quick), (the, brown)  |
| [The **quick** brown fox] jumps over the lazy dog. | quick       | the, brown, fox           | ...                         |
| [The quick **brown** fox jumps] over the lazy dog. | brown       | the, quick, fox, jumps    | ...                         |
| The [quick brown **fox** jumps over] the lazy dog. | fox         | quick, brown, jumps, over | ...                         |
| ...                                                | ...         | ...                       | ...                         |

**Model Details**

<img src="{{ '/assets/imgs/Pre-Training-in-NLP/1.png' | relative_url }}" style="zoom:60%;">

*(图源:[Lilian Weng. Learning Word Embedding](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html)[^3],我想这是我写笔记为止见到的非常详细清楚的一张图了!)*

输入的$x$是一个$V$维one-hot编码，即只有一个位置为1；接下来与embedding matrix做矩阵相乘，选中embedding matrix中的对应$i$行，得到一个$N$维向量，此步代表了对原单词的编码；接下来是输出层，我们将$N$维向量与context matrix相乘，context matrix的$j$列可以看作是第$j$个单词对应当前target word的权重，相乘得到$\hat{y}$ 为$V$维softmax概率，即对应位置单词为target word上下文的概率；对于每个训练样本，预期的输出$y$是单词上下文的one-hot编码。

训练完成后的hidden layer中的embedding matrix存储了我们所有单词的对应编码；

该基础的网络模型在字典 (字典包括了所有的单词)极具增大时工作的不尽如人意，假设我们有10, 000个单词，每个单词有$N = 300$特征。网络模型包括两层：隐藏层和输出层，每层都将有$300 \times 10, 000$个权重！在这么大的神经网络上运行梯度下降是很慢的，更糟糕的是，需要大量的训练数据来调整这么多的权重，避免过度拟合，数百万的权重乘以数十亿的训练样本，训练该模型将是一个灾难。

Word2Vec的作者在他们的第二篇论文[^4]中通过以下创新尝试解决这些问题：

- `软滑动窗口`。意在对较远的上下文分配较少的权重，一种启发式方法是，设定最大窗口大小$s_{max}$，在抽取训练样本的时候窗口大小在$[1, s_{max}]$中随机产生；
- `对频繁出现的词进行子采样`。极为频繁的词可能过于笼统，无法区分上下文，另一方面，稀有词更有可能携带独特的信息 (在信息学中的信息编码中有类似的思想)。为了平衡频繁出现的词语和罕见的词语，Mikolov等人建议以一定的概率$1-\sqrt{t/f(w)} $ 丢弃词语；
- `“负采样”`。值得注意的一点是，输入以及预期的输出都是one-hot编码，假设目前的训练样本为 (the, brown)，在后向传播的过程中，我们将提高context matrix中"brown"对应列的权重，降低其他列的权重，只需要调整embedding matrix的"the"对应行权重。负采样提出为了减少需要调整的参数数目，在更新context matrix时，我们提高"brown"对应列的权重，但仅对部分其他列权重进行降低，实验过程中这部分列数目在5~20。负采样按照如下公式选择那些需要降低权重的列：$P(w_i) = \frac{f(w_i)^(3/4)}{\sum_{i=0}^n f(w_j)^{3/4} }$，其中$f(w_i)$代表该单词出现次数；
- `先学短语`。一个短语往往是作为一个概念单位，而不是单个词的简单组成。例如，即使我们知道 "new "和 "york "的含义，我们也不能真正知道 "New York "是一个城市名称。在训练单词嵌入模型之前，先学习这样的短语并将其作为单词单元，可以提高结果的质量。学习短语的一种方式是：$s_{phrase} = \frac{C(w_i,w_j) - \delta }{C(w_i)C(w_j)}$，其中$C(.)$代表计数，$\delta$是阈值，防止学到超低频短语，分数$s$越高，表示成为短语的概率越大，要形成长于两个单词的短语，可以多次执行学习短语程序。

值得注意的是，对频繁出现的词进行子采样和应用负采样等不仅减少了训练过程的计算负担，而且还提高了我们得到的词向量的质量。

## ELMo

NLP领域中的预训练思路可以分为两类：

第一代预训练模型专注于word embedding的学习(word2vec)，神经网络本身关于特定任务的部分参数并不是重点。其特点是`context-free`，也即`word embedding`，每个token的表示与上下文无关，比如“苹果”这个词在分别表示水果和公司时，对应的word embedding是同样的，其中具有代表性的有Word2vec，GloVe等。

第二代预训练模型以`context-aware`为核心特征，单词的编码不是固定的，而是做为输入的 (包含该单词的) 句子的函数，也就是说“苹果”这个词在分别表示水果和公司时，对应的output是不一样的，其中具有代表性的有ELMo，GPT，BERT等。

在本节，我们将简要介绍ELMo的模型基本结构，半监督训练方法等。

**Model Details**

<img src="{{ '/assets/imgs/Pre-Training-in-NLP/2.png' | relative_url }}" style="zoom:60%;">

ELMo分为



## GPT



## Bert











需要提一点的是，早期的研究者们在模型结构上做的尝试比较多，比如ELMo使用了双向LSTM。然而在Transformer出现后，研究者们研究的重点就从模型结构转移到了训练策略上。比如GPT和BERT都是基于Transformer结构的: GPT基于Transformer decoder，而BERT基于Transformer encoder。



[^1]:[Word2Vec Tutorial-The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) - 11 Jan 2017 by Chris McCormick
[^2]: Mikolov, Tomas, Kai Chen, Greg Corrado, and Jeffrey Dean. "[Efficient estimation of word representations in vector space.](https://arxiv.org/abs/1301.3781)" *arXiv preprint arXiv:1301.3781* (2013).
[^3]: [Learning Word Embedding](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html) - Oct 15, 2017 by Lilian Weng
[^4]: Mikolov, Tomas, et al. "[Distributed representations of words and phrases and their compositionality.](https://arxiv.org/pdf/1310.4546.pdf)" *Advances in neural information processing systems*. 2013.





























