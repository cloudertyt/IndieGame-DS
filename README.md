# DS4CI Portfolio — Steam 独立游戏市场全链路数据科学分析

> 课程：Data Science for Creative Industries (DS4CI) 2025-26
> 作者：Tyt13
> 数据源：Steam 历史数据集（`steam.csv`，约 27,000 款游戏）+ Steam Web API 模拟数据

---

## 项目概览

本 Portfolio 以**独立游戏创业决策**为核心业务场景，跨越三份 Notebook 构建了一条完整的数据科学分析链路：

```
Notebook 1 — 宏观市场基线
    ↓  (从"感知市场"到"理解受众")
Notebook 2 — 微观玩家行为与聚类
    ↓  (从"诊断性分析"到"预测性与处方性分析")
Notebook 3 — 商业成功率预测、推荐系统与算法正义批判
```

每份 Notebook 对应特定课程周次，将课堂方法论迁移至独立游戏数据场景，并在每个模块末尾提供批判性的伦理与社会反思。

---

## 文件结构

```
Portfolio/
├── README.md                   ← 本文档
├── 1/
│   ├── 1_Games.ipynb           ← Notebook 1：宏观市场分析（15 模块）
│   └── Data/
│       └── steam.csv           ← 主数据集（需自行放置）
├── 2/
│   └── 2_Games.ipynb           ← Notebook 2：微观玩家行为分析（10 模块）
└── 3/
    └── 3_Gamed.ipynb           ← Notebook 3：预测建模与推荐系统（13 模块）
```

---

## 环境依赖

```bash
# 核心依赖
pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install vaderSentiment networkx shap imbalanced-learn
```

| 库 | 用途 |
|---|---|
| `pandas` / `numpy` | 数据清洗、特征工程 |
| `matplotlib` / `seaborn` | 所有可视化 |
| `scipy.stats` | T 检验、ANOVA、Pearson 相关、Wilson 置信区间 |
| `sklearn` | 线性回归、随机森林、GridSearchCV、TF-IDF、TruncatedSVD、SMOTE |
| `vaderSentiment` | NLP 情感分析（Module 15 in NB1、Module 10 in NB2） |
| `networkx` | 微标签共现网络拓扑（NB3 Module 2） |
| `shap` | 可解释 AI / 特征归因（NB3 Module 9） |
| `imbalanced-learn` | SMOTE 过采样（NB3 Module 4） |

---

## Notebook 1 — Steam 独立游戏市场数据分析与创业启示

**文件：** `1/1_Games.ipynb`
**对应课程周次：** Week 1（EDA 基础）、Week 2（数据清洗与描述性统计）、Week 5（线性回归）

### 研究问题

> 对于资源有限的独立游戏团队，Steam 历史数据能否为立项决策提供可量化的风险评估框架？

### 数据

- **数据集：** `Data/steam.csv`（约 27,000 款游戏，含价格、好评数、游玩时长、发行商、平台、标签等 18 个字段）
- **分析子集：** 包含 `Indie` 标签的游戏（约 19,400 款）
- **派生特征：** `revenue_score = price × positive_ratings`、`core_gameplay`（首个非 Indie 类型标签）、`ratings_zscore`、`wilson_score`

### 模块索引

| # | 模块名称 | 核心方法 | 关键发现 |
|---|---------|---------|---------|
| 1 | 数据加载与独立游戏切片 | `dropna()`、`str.contains()`、子集切片 | 独立游戏占全平台约 71% |
| 2 | 描述性统计：独立游戏基准线 | `describe()`、中位数 vs 均值对比 | 好评数中位数远低于均值，幸存者偏差显著 |
| 3 | 分组基准线：细分玩法赛道 | `groupby()`、`apply()` 提取核心类型 | 动作、冒险、策略赛道最拥挤 |
| 4 | 定价策略可视化（线性 vs 对数刻度）+ 时间序列演变 | 双轴折线图、`twinx()`，log scale scatter | 对数刻度揭示隐藏相关；定价中位数 2014 年后持续下滑 |
| 5 | 数据归一化与爆款拆解 | Z-Score 标准化，阈值 > 5 | 19,400 款中仅 51 款达到现象级（Z > 5） |
| 6 | 不确定性量化：品类立项风险评估 | SEM 置信区间柱状图 | RPG / 模拟类 SEM 最大，"高风险高回报" |
| 7 | 大型发行商分布基准线 | `value_counts()`、`sum().sort_values()` | 头部发行商呈现长尾效应 |
| 8 | 发行商表现趋势可视化 | 自定义多线折线图函数，log scale 可选 | 头部大厂年度好评数趋势差异显著 |
| 9 | 跨平台发行策略分析 | 平台分类函数、`str.split()`，双轴柱状图 | 全平台（Win+Mac+Linux）定价溢价约 18%，但销量红利存在 |
| 10 | 微标签（Micro-Tags）市场可能性拆解 | `explode()`、过滤宽泛标签、营收中位数排名 | "Roguelike"、"Metroidvania" 等机制标签中位营收显著高于大类 |
| 11 | 产品特性 ROI 评估 | 二元特征提取、中位数对比柱状图 | 合作模式、手柄支持对基础销量有统计显著正向贡献 |
| 12 | 回归建模：量化营收驱动因素 | `LinearRegression`、`log(revenue_score)`、`r2_score` | 本地化语言数量是最强正向回归系数 |
| 13 | 重构评价指标：威尔逊区间 | Wilson Score 公式、置信区间下限排名 | 解决高评价数低好评率 vs 低评价数高好评率的排序问题 |
| 14 | 玩家留存与游戏循环健康度检测 | 均值/中位数游玩时长比值、散点图 | 大量独立游戏中位游玩时长低于 120 分钟退款红线 |
| 15 | NLP 延伸：游戏标签情感基调分析 | VADER、Pearson 相关、ANOVA、年度线性回归趋势 | 情感得分与 log(revenue) 相关性弱（r ≈ 0.05），但赛道间 ANOVA 显著 |

### 核心结论

- **立项基准：** 用品类**中位数**而非均值做流水预估，防止被爆款数据误导
- **赛道选择：** 微标签组合（如"Roguelike + Deckbuilder"）优于宽泛大类
- **功能优先级：** 合作模式、手柄支持、本地化语言数量 ROI 最高
- **核心循环：** 确保首两小时内容密度，规避 Steam 退款红线

---

## Notebook 2 — 独立游戏微观受众洞察：玩家留存周期、区域定价与用户画像聚类分析

**文件：** `2/2_Games.ipynb`
**对应课程周次：** Week 3（卡方检验与分类数据统计）、Week 4（T 检验与 ANOVA）、Week 6（K-Means 聚类）

### 研究问题

> 在从宏观市场转向微观玩家行为的尺度下，如何通过玩家行为聚类构建运营决策的受众画像？

### 数据

- **数据来源：** Steam Web API 接口模拟（受限于 API 访问速率与隐私政策，以统计学等价的合成数据集呈现）
- **数据集：** `df_players`（N = 800 名模拟玩家，含游玩时长、成就解锁率、购买渠道、地理区域、硬件配置等特征）

### 模块索引

| # | 模块名称 | 核心方法 | 关键发现 |
|---|---------|---------|---------|
| 1 | 实时数据获取与结构化 | Steam API 模拟管道、JSON 解析、DataFrame 构建 | 建立从非结构化 API 响应到特征矩阵的标准化流程 |
| 2 | 游玩时长分布形态学分析 | KDE 曲线、对数变换、偏度分析 | 游玩时长呈严重右偏，对数变换后接近正态 |
| 3 | 留存漏斗与成就衰减曲线 | `explode()` 展开成就里程碑、漏斗图 | 成就完成率在首个"关卡门槛"处出现最大跌幅（留存漏斗）|
| 4 | 购买决策对参与度影响 | 独立样本 T 检验（`ttest_ind`）、小提琴图 | 折扣购买者游玩时长中位数显著低于原价购买者（Week 4 方法论） |
| 5 | 硬件性能分层与技术资产评估 | ANOVA（`f_oneway`）、箱线图分组 | 高配硬件玩家成就解锁率显著更高，支持硬件分层定价策略 |
| 6 | 地理分布与购买力平价倒挂检验 | 卡方检验（`chi2_contingency`）、地区分组 | 部分新兴市场区域存在购买力平价定价倒挂（Week 3 方法论） |
| 7 | 社交机制偏好度验证 | 多人 vs 单人游玩时长 T 检验、柱状图 | 支持联机功能的游戏平均游玩时长高出约 35% |
| 8 | 核心指标多重共线性扫描 | Pearson 相关热图、VIF 检测 | 游玩时长与成就数高度相关（r > 0.8），回归前需处理共线性 |
| 9 | K-Means 构建多维玩家画像 | `StandardScaler`、肘部法则、K-Means、PCA 降维可视化 | 最优 K = 3：折扣买家、核心玩家、成就猎人 三类画像（Week 6 方法论）|
| 10 | NLP 延伸：玩家群体情感标签画像 | VADER 情感分析（连接 NB1 Module 15）、ANOVA 跨聚类差异 | 核心玩家聚类与强正面情感标签游戏的游玩时长相关性最强 |

### 核心结论

- **三类玩家画像：** ①折扣敏感型（低游玩时长、低成就率）②核心参与型（高时长、高成就）③成就猎人型（高成就率但时长中等）
- **运营优先级：** 面向"核心参与型"玩家设计 DLC 和成就系统 ROI 最高
- **定价策略：** 频繁打折会吸引参与度低的玩家，稀释核心玩家社区质量
- **跨 Notebook 衔接：** NB1 宏观基线 + NB2 微观聚类 = 立项决策的双层数据支撑

---

## Notebook 3 — 独立游戏商业成功率预测、推荐系统原型与算法正义批判

**文件：** `3/3_Gamed.ipynb`
**对应课程周次：** Week 7（内容推荐系统）、Week 8（协同过滤与嵌入层）

### 研究问题

> 如何在缺乏历史用户数据的情况下（冷启动问题），为新发售独立游戏预测商业成功概率并提供精准推荐？

### 数据

- **数据集：** 合成游戏特征数据集（N = 8,000，含标签向量、价格、好评率、游玩时长、平台支持等特征）
- **目标变量：** `is_success`（`revenue_score` 中位数以上 = 1，二分类）

### 模块索引

| # | 模块名称 | 核心方法 | 关键发现 |
|---|---------|---------|---------|
| 1 | 特征工程与目标变量定义 | 合成数据生成、二分类目标、`revenue_score` 分位数切割 | 约 50% 的游戏达到"商业成功"阈值（平衡分类基准） |
| 2 | 微标签网络拓扑分析 | `NetworkX` 共现图、Betweenness Centrality 排名 | "Roguelike"、"RPG" 是最高中介中心性节点（市场连接器） |
| 3 | 文本特征空间降维 | `TfidfVectorizer(max_features=500)`、`TruncatedSVD(n_components=min(50, n_features-1))`、累计解释方差曲线 | 前 15 个 SVD 主成分捕获约 85% 的标签语义方差 |
| 4 | 非平衡数据集算法重采样 | SMOTE 过采样、类别分布对比可视化 | SMOTE 使少数类从约 40% 提升至 50%，改善分类器偏向 |
| 5 | 随机森林分类器训练 | `RandomForestClassifier`、`StratifiedKFold`、`classification_report` | 基础随机森林准确率约 72% |
| 6 | 超参数网格搜索与交叉验证 | `GridSearchCV`（`n_estimators`、`max_depth`、`min_samples_split`）| 最优参数组合使 F1 提升约 3-5 个百分点 |
| 7 | 预测性能指标多维评估 | 混淆矩阵、Precision/Recall/F1、指标柱状图 | 最优模型 F1 ≈ 0.74，Recall 优先以避免"漏判潜力爆款" |
| 8 | ROC-AUC 曲线分析 | `roc_curve`、AUC 面积、Youden's J 最优阈值 | AUC ≈ 0.80，Youden's J 最优阈值约 0.42（非默认 0.5）|
| 9 | 可解释 AI：SHAP 值特征归因 | `shap.TreeExplainer`、SHAP beeswarm 图、SHAP API 兼容处理 | 正面好评率和游玩时长是最强正向预测特征 |
| 10 | 推荐系统冷启动问题定义 | 协同过滤 vs 内容推荐对比图 | 新游戏无历史交互记录，协同过滤失效，必须使用内容推荐 |
| 11 | 内容推荐：TF-IDF 文本向量化 | `TfidfVectorizer`（5 款参考游戏语料库）、词频矩阵展示 | TF-IDF 能有效区分"Roguelike vs JRPG vs Metroidvania"的词频特征 |
| 12 | 余弦相似度计算与推荐输出 | `cosine_similarity`、相似度矩阵热图、Top-N 推荐 | Hollow Knight 与 Ori 相似度最高（0.73），语义契合 |
| 13 | 基于聚类的 A/B 测试实验设计模拟 | `scipy.stats.ttest_ind`、对照组 vs 实验组、双轴柱状图 | 内容推荐组点击率显著高于随机推荐（p < 0.05） |

### 核心结论

- **商业成功预测：** 正面好评率、游玩时长、合作模式支持是最强预测特征（SHAP 验证）
- **最优阈值：** 使用 Youden's J 而非默认 0.5，可减少潜力游戏的漏判率
- **推荐系统：** 冷启动场景下 TF-IDF + 余弦相似度优于协同过滤；A/B 测试统计显著验证其转化率增益
- **算法正义批判：** 随机森林和 SHAP 均以历史 Steam 数据为训练基础，对非英语文化背景游戏存在系统性偏见

---

## 跨 Notebook 方法论归纳

| 课程方法论 | NB1 应用 | NB2 应用 | NB3 应用 |
|-----------|---------|---------|---------|
| EDA + 描述统计 | Module 1-2 | Module 1-2 | Module 1 |
| 分组聚合与基准线 | Module 3, 7 | Module 5-7 | — |
| 对数变换与可视化 | Module 4, 8 | Module 2 | — |
| 统计检验（T/ANOVA/卡方）| — | Module 4, 5, 6 | Module 13 (t-test) |
| 线性回归 | Module 12 (+ NLP 趋势) | — | — |
| 无监督聚类（K-Means） | — | Module 9 | — |
| 降维（SVD / PCA） | — | Module 9 (PCA) | Module 3 (SVD) |
| 内容推荐（TF-IDF + 余弦） | — | — | Module 11-12 |
| 随机森林 + 超参调优 | — | — | Module 5-6 |
| 可解释 AI（SHAP） | — | — | Module 9 |
| NLP 情感分析（VADER） | Module 15 | Module 10 | — |
| A/B 测试 | — | — | Module 13 |

---

## 数据伦理与社会批判

三份报告均在结语章节对以下问题进行了批判性反思：

1. **幸存者偏差（Survivorship Bias）：** Steam 数据集仅包含已发行游戏，系统性低估了失败项目的规律
2. **文化与语言偏见：** 威尔逊区间、回归模型和 SHAP 值均以英语主导的西方市场历史数据为基础，对非英语文化圈游戏存在系统性低估
3. **数字碳足迹：** 鼓励无限延长游玩时长的优化目标与气候正义存在张力——内容质量与碳效率的双重优化应成为行业未来议题
4. **算法推荐的马太效应：** 内容推荐系统的余弦相似度偏向"与已成功游戏相似"的项目，可能抑制真正创新性的独立作品获得曝光

---

## 如何运行

```bash

# 1. 启动 Jupyter
jupyter notebook

# 2. 按顺序运行（NB1 → NB2 → NB3）
# 每个 Notebook 均可独立运行，无跨文件依赖
# NB1 需要 Data/steam.csv（从 1/ 目录下运行）

# 3. 或使用 nbconvert 批量执行
cd Portfolio/1 && jupyter nbconvert --to notebook --execute 1_Games.ipynb
cd Portfolio/2 && jupyter nbconvert --to notebook --execute 2_Games.ipynb
cd Portfolio/3 && jupyter nbconvert --to notebook --execute 3_Gamed.ipynb
```

> **注意：** NB2 和 NB3 使用合成数据集，无需额外数据文件，开箱即运行。

---

*Portfolio by Tyt13 — DS4CI 2025-26*
