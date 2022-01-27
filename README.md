# gnn-study

用于记录自学 **图神经网络**。

## 目录

- [目录](#目录)
- [图的基础知识](#图的基础知识)
- [Graph Embedding](#Graph-Embedding)
	- [DeepWalk](#DeepWalk)
	- [LINE](#Line-Large-scale-Information-Network-Embedding)
	- [node2vec](#node2vec)
	- [struc2vec](#struc2vec)
	- [SDNE](#SDNE)
- [Graph Neural Network](#Graph-Neural-Network)
	- [GCN](#GCN) 
	- [GraphSAGE](#GraphSAGE)
- [参考链接](#参考链接)

## 图的基础知识

**图**

一张图是一个二元组 $G = (V, E)$，其中集合 $V$ 中的元素称为 **节点**，集合 $E$ 中的元素是两个节点组成的无序对，称为 **边**。

**图的分类**

有向图，无向图，混合图（边既可以有向也可以无向）。

**图的表示**

邻接表，邻接矩阵。

**度**

入度（箭头指入），出度（箭头指出）。

**子图**

对于图 $G$ 和图 $G^{'}$，若 $V(G^{'}) \subset V(G)$ 且 $E(G^{'}) \subset E(G)$，则称 $G^{'}$ 是 $G$ 的子图。

**连通图**

对于一个 **无向图**，如果任意的节点 $i$ 能够通过一些边到达节点 $j$，则称为连通图。

**有向图连通性**

- **强连通图**：任意两个节点互相对称可达的 **有向图**。
- **弱连通图**：任意两个节点至少单向可达的 **有向图**。

**图直径**

图中任意两点之间 **最短路径** 中的 **最长者**。

**度中心性 Degree Centrality**

用来刻画节点中心性的最直接度量，用来发现图中与其他关联最多的顶点。

$$
\text{norm(degree)} = \frac{\text{degree}}{N - 1}
$$

**中介中心性 Betweenness Centrality**

衡量顶点在图中担任 **桥梁** 角色的程度，即顶点出现在其他任意两个顶点对之间最短路径的次数。

$$
\text{betweenness} = \frac{经过该节点的最短路径次数}{C^2_N}
$$

**接近中心性 Closeness Centrality**

用于计算每个顶点到其他所有顶点的最短距离之和。

$$
\text{closeness} = \frac{n - 1}{节点到其他节点最短路径之和}
$$


**特征向量中心性**

测量节点对网络影响的方式。给定一个节点集合为 $|V|$ 的图 $G = (V, E)$，定义其 **邻接矩阵** 为 $A = (a_{v, t})$，当 $v$ 与 $t$ 相连时有 $a_{v, t} = 1$，否则 $a_{v, t} = 0$。则节点 $v$ 中心性 $x$ 的分数求解公式为：

$$
x_v = \frac{1}{\lambda}\sum_{t \in G}a_{v, t}x_t
$$

## Graph Embedding

### DeepWalk

> 主要思想是在图结构上进行 **随机游走**，产生大量 **节点序列**，然后将这些序列作为训练样本输入 **word2vec** 进行训练，采用 **skip-gram** 算法，最大化随机游走序列的似然概率，并使用 **随机梯度下降** 学习参数，得到物品的 **embedding**。

### LINE: Large-scale Information Network Embedding

> 捕获节点的一阶和二阶相似度，分别求解，再将一阶二阶拼接在一起，作为节点的 embedding。

#### 一阶相似度

用于描述图中成对顶点之间的局部相似度。形式化描述为若 $u, v$ 之间存在直连边，则边权 $w_{uv}$ 即为两个顶点的相似度，若不存在直连边，则一阶相似度为 0。**仅能用于无向图**。

对于每一条无向边 $(i, j)$，顶点 $v_i, v_j$ 之间的 **联合概率** 是。其中，$\vec{u}_i$ 为顶点 $v_i$ 的低维向量表示。

$$
p_1 = \frac{1}{1 + \exp(-\vec{u}_i^T \cdot \vec{u}_j)}
$$

**经验分布** 为：

$$
\hat{p_1} = \frac{w_{ij}}{\sum_{(i, j) \in E} w_{ij}}
$$

常用指标是 **KL 散度**，则 **优化目标** 为 **最小化**：

$$
O_1 = d(\hat{p_1}(\cdot, \cdot), p_1(\cdot, \cdot)) = -\sum_{(i, j) \in E} w_{ij} \log p_1(v_i, v_j)
$$

#### 二阶相似度

用来表示 $u$ 与 $v$ 之间邻居节点相似情况。**可以用作有向图**。

对于有向边 $(i ,j)$，定义给定顶点 $v_i$ 条件下，产生上下文（邻居）顶点 $v_j$ 的概率是。其中，$|V|$ 是上下文顶点的个数。

$$
p_2(v_j | v_i) = \frac{\exp(\vec{u}_j^T \cdot \vec{u}_i)}{\sum_{k = 1}^{|V|} \exp(\vec{u}_k^T \cdot \vec{u}_i)}
$$

**优化目标** 是。其中 $\lambda_i$ 是控制节点重要性的因子，可以通过顶点的度数或 PageRank 等方法估计得到。

$$
O_2 = \sum_{i \in V} \lambda_i d(\hat{p}_2(\cdot | v_i), p_2(\cdot | v_i)) 
$$

**经验分布** 是。其中 $d_i$ 是顶点 $v_i$ 的 **出度**。

$$
\hat{p}_2 = (v_j | v_i) = \frac{w_{ij}}{d_i}
$$

使用 **KL 散度** 并且设 $\lambda_i = d_i$，忽略常数项，有：

$$
O_2 = - \sum_{(i, j) \in E} w_{ij} \log p_2(v_j | v_i)
$$

### Node2Vec

> 采用 **有偏向的随机游走** 策略来获取顶点的近邻序列，使用类似 skip-gram 方式生成节点 embedding。

给定当前顶点 $v$，访问下一个顶点 $x$ 的概率是。其中 $\pi_{vx}$ 是顶点 $v$ 和顶点 $x$ 之间的未归一化转移概率，$Z$ 是归一化常数。

$$
P(c_i = x | c_{i - 1} = v) =
\begin{cases}
\frac{\pi_{vx}}{Z} & \text{if} \ (v, x) \in E \\
0 & \text{otherwise} \\
\end{cases}
$$

引入两个超参数 $p$ 和 $q$ 来控制随机游走的策略，假设当前随机游走经过边 $(t, v)$ 到达顶点 $v$，设 $\pi_{vx} = \alpha_{pq}(t, x) \cdot w_{vx}$，$w_{vx}$ 是顶点 $v$ 和顶点 $x$ 之间的边权。

$$
\alpha_{pq}(t, x) = 
\begin{cases}
\frac{1}{p} & \text{if} \ d_{tx} = 0 \\
1 & \text{if} \ d_{tx} = 1 \\
\frac{1}{q} & \text{if} \ d_{tx} = 2 \\
\end{cases}
$$

$d_{tx}$ 是顶点 $t$ 和顶点 $x$ 之间的最短路径距离。

- $q$ 值小，则 **探索性强**。会捕获 **同质性节点**，即相邻接点表示相似，更像 **DFS**。
- $p$ 值小，则 **相对保守**。会捕获 **结构性**，即某些节点的图上结构，更像 **BFS**。

### Struc2Vec

> 通过比较两个顶点间距离为 $k$ 的环路上的有序度序列来层次化衡量结构相似度，对图的结构信息进行捕获，从而发掘节点间 **空间结构相似性**。当图的结构重要性大于邻居重要性时，有较好的效果。

- $R_k(u)$：到顶点 $u$ 距离为 $k$ 的 **顶点集合**。例如 $R_0(u)$ 即节点本身组成的集合，$R_1(u)$ 则是直接与节点相邻节点组成的集合。
- $s(S)$：顶点集合 $S$ 的 **有序度序列**。
- $f_k(u, v)$：顶点 $u$ 和顶点 $v$ 之间距离为 $k$ 的环路上的结构距离。

$$
f_k(u, v) = f_{k - 1}(u, v) + g(s(R_k(u)), s(R_k(v)))
$$

其中 $g(D_1, D_2) \geq 0$ 是衡量有序度序列 $D_1$ 和 $D_2$ 的距离函数，且 $f_{-1} = 0$。

- **Dynamic Time Warping (DTW)**：用来衡量两个不同长度且含有重复元素的序列的距离的算法。
- $d(a, b) = \frac{\max(a, b)}{\min(a, b)} - 1$：基于 **DTW** 的元素之间距离函数。
- $w_k(u ,v) = e^{-f_k(u ,v)}$：第 $k$ 层中顶点 $u$ 和 顶点 $v$ 的 **边权**。

通过 **有向边** 将 **属于不同层次的同一顶点** 连接起来，则不同层次同一顶点间的边权是。

$$
\begin{align}
w(u_k, u_{k + 1}) &= \log(\Gamma_k(u) + e) \\
w(u_k, u_{k - 1}) &= 1 \\
\end{align}
$$

- 采用 **随机游走**，每次采样的顶点更倾向于选择与当前节点结构相似的节点。

若决定在当前层游走，则从顶点 $u$ 到顶点 $v$ 的概率是。

$$
p_k(u, v) = \frac{e^{-f_k(u, v)}}{\sum_{v \in V, v \neq u} e^{-f_k(u, v)}}
$$

若决定切换不同的层，则以如下概率选择 $k + 1$ 层或 $k - 1$ 层。

$$
\begin{align}
p_k(u_k, u_{k + 1}) &= \frac{w(u_k, u_{k + 1})}{w(u_k, u_{k + 1} + w(u_k, u_{k - 1}))} \\
p_k(u_k, u_{k - 1}) &= 1 - p_k(u_k, u_k + 1) \\
\end{align}
$$

### SDNE

> 采用多个非线性层的方式捕获一阶二阶的相似性。

## Graph Neural Network

### GCN

> 多层的图卷积神经网络，每一个卷积层仅处理一阶邻域信息，通过叠加若干卷积层可以实现多阶邻域的信息传递。

每一个 **卷积层的传播规则**是。

$$
H^{(l + 1)} = \sigma(\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})
$$

- $\hat{A} = A + I_N$：无向图 $G$ 的邻接矩阵加上自连接，$I_N$ 是单位矩阵。
- $\hat{D}$ 是 $\hat{A}$ 的度矩阵，即 $\hat{D}_{ii} = \sum_j\hat{A}_{ij}$。
- $H^{(l)}$ 是第 $I$ 层的激活单元矩阵，$H^0 = X$。
- $W^{(l)}$ 是每一层的参数矩阵。

### GraphSAGE

> 通过学习一个对邻居节点进行聚合表示的函数来产生目标顶点的 embedding 向量。先采样邻居节点的向量信息，然后对这些信息进行聚合操作，最后和自己的信息进行拼接得到单个向量，并且 **训练权重函数** 来生成节点的 embedding。

#### 采样邻居节点

采样数量 **少于** 邻居节点数时，采用 **有放回** 抽样方法。采样数量 **多于** 邻居节点数时，采用 **无放回** 抽样方法。

**minibatch** 仅采样部分邻居节点，不使用全图信息，适用于大规模图训练。具体方式如下图：

![graphsage-minibatch](./README/graphsage-minibatch.png)

#### 聚合拼接

$$
\begin{align}
h^k_{N(v)} & \leftarrow \text{AGGREGATE}_k(\{h^{k - 1}_u, \forall u \in N(v) \}) \\
h^k_v & \leftarrow \sigma(W^k \cdot \text{CONCAT}(h^{k - 1}_v, h^k_{N(v)})) \\
\end{align}
$$

**常用聚合函数**

- **MEAN aggregator**：将目标顶点和邻居顶点的 $k - 1$ 层向量 **拼接** 起来，然后对向量的每个维度进行 **求均值** 操作，将得到的结果做一次 **非线性变化** 产生目标顶点的第 $k$ 层表示向量。

$$
h^k_v \leftarrow \sigma(W \cdot \text{MEAN}(\{h^{k - 1}_v\} \cup \{h^{k - 1}_u, \forall u \in N(v) \}))
$$

- **Pooling aggregator**：先对目标顶点的邻结点表示向量进行一次非线性变换，之后再进行一次 pooling 操作（maxpooling 或 meanpooling），将得到结果与目标顶点的表示向量拼接，最后再进行一次非线性变换得到目标顶点的第 $k$ 层表示向量。

$$
\text{AGGREGATE}^{\text{pool}}k = \max(\{\sigma (W_{\text{pool}} h^k_{u_i} + b), \forall u_i \in N(v) \})
$$

- **LSTM aggregator**：有更强的表达能力，但不是输入对称的，所以使用时需要对顶点的邻句 **乱序**。

#### 参数学习

无监督学习。基于图的损失函数希望 **临近的节点具有相似的向量表示**，同时让 **分离的节点表示尽可能区分**。目标函数如下：

$$
J_G(z_u) = -\log (\sigma (z_u^T z_v)) - Q \cdot E_{v_n \sim P_n(v)} \log(\sigma (-z_u^T z_{v_n}))
$$

其中，$v$ 是通过 **固定长度的随机游走** 出现在 $u$ 附近的节点，$p_n$ 是负采样的概率分布，$Q$ 是负样本数量。

### GAT

$$
\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\vec{a}^T[W \cdot \vec{h}_i \| W \cdot \vec{h}_j])}{\sum_{k \in \N_i} \exp(\text{LeajyReLU}(\vec{a}^T [W \cdot \vec{h}_i \| W \cdot \vec{h}_k])}))
$$

$$
\vec{h^{'}_i} = \sigma(\sum_{j \in N_i} \alpha_{ij} \cdot W \cdot \vec{h_j})
$$

$W$ 为 **训练的权重**，$\|$ 表示将两个向量 **拼接** 在一起。即我们先求出目标节点和某邻居节点之间的 $\alpha$ 系数，然后对目标节点的所有邻居节点进行 **特征结合**，迭代求出下一轮目标节点的表示向量。

**多头注意力机制** 使用不同参数重复训练多次拼接成一个大矩阵。

$$
\vec{h^{'}_i} = \|^K_{k = 1} \sigma(\sum_{j \in N_i} \alpha_{ij}^k \cdot W^k \cdot \vec{h_j})
$$

在最后的预测层则直接在向量的每个位置 **取平均值** 得到节点的最终特征。

$$
\vec{h^{'}_i} = \sigma(\frac{1}{K} \sum^K_{k = 1}\sum_{j \in N_i} \alpha_{ij}^k \cdot W^k \cdot \vec{h_j})
$$

## 参考链接

[零基础多图详解图神经网络（GNN/GCN）](https://www.youtube.com/watch?v=sejA2PtCITw)

[【图神经网络】GNN从入门到精通](https://www.bilibili.com/video/BV1K5411H7EQ?p=2)

