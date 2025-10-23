import pykeen.pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator

# Step 1: 数据加载
# 定义训练集和测试集路径
train_path = 'train.txt'
test_path = 'test.txt'

# 加载训练集和测试集
training_triples = TriplesFactory.from_path(train_path, create_inverse_triples=True)
testing_triples = TriplesFactory.from_path(test_path)

# Step 2: 使用 ComplEx 模型进行训练
pipeline_result = pykeen.pipeline.pipeline(
    model='ComplEx',  # 使用 ComplEx 模型
    training=training_triples,
    testing=testing_triples,
    model_kwargs={
        'embedding_dim': 100,  # 嵌入向量的维度
    },
    optimizer='Adam',  # 优化器
    optimizer_kwargs={
        'lr': 0.01,  # 学习率
    },
    loss='SoftplusLoss',  # 损失函数
    training_kwargs={
        'num_epochs': 100,  # 训练轮数
        'batch_size': 128,  # 批量大小
    },
    random_seed=42,  # 随机种子，确保实验可复现
    device='cpu',  # 设备，可切换为 'cuda' 使用 GPU
)

# Step 3: 模型评估
evaluator = RankBasedEvaluator()
results = evaluator.evaluate(
    model=pipeline_result.model,
    mapped_triples=testing_triples.mapped_triples,
    additional_filter_triples=[training_triples.mapped_triples],
)
print("Evaluation Results:", results)

# Step 4: 保存嵌入
entity_embeddings = pipeline_result.model.entity_representations[0]
relation_embeddings = pipeline_result.model.relation_representations[0]

# 保存实体和关系嵌入为 numpy 文件
import numpy as np
np.save("complex_entity_embeddings.npy", entity_embeddings.cpu().detach().numpy())
np.save("complex_relation_embeddings.npy", relation_embeddings.cpu().detach().numpy())

# 可视化嵌入维度和统计信息
print("Entity Embedding Shape:", entity_embeddings.shape)
print("Relation Embedding Shape:", relation_embeddings.shape)
