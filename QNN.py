"""
🧠 QNN Triple Classifier 模块
Author: DOCTOR + 歌蕾蒂娅（2025）

功能：
- 使用量子神经网络（QNN）对三元组进行分类（ontology vs instance）
- 支持训练、自定义训练样本、误差输出
- 与量子设备模拟器（PennyLane）配合运行
"""

import pennylane as qml
from pennylane import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

class QuantumClassifier:
    def __init__(self, n_qubits=4, n_layers=2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev)
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        self.weights = 0.01 * np.random.randn(n_layers, n_qubits, 3)

    def predict(self, x: List[float]) -> int:
        if len(x) != self.n_qubits:
            raise ValueError(f"输入维度必须为 {self.n_qubits}")
        x = np.array(x, requires_grad=False)
        output = self.circuit(x, self.weights)
        return int(output < 0)  # 可扩展为多分类

    def loss(self, X: List[List[float]], Y: List[int]) -> float:
        loss = 0
        for x, y in zip(X, Y):
            pred = self.circuit(np.array(x), self.weights)
            loss += (pred - (1 - 2 * y)) ** 2
        return loss / len(X)

    def train(self, X: List[List[float]], Y: List[int], epochs=100, lr=0.1):
        opt = qml.GradientDescentOptimizer(lr)
        for i in range(epochs):
            self.weights = opt.step(lambda w: self.loss(X, Y), self.weights)
            if (i + 1) % 10 == 0:
                print(f"Epoch {i+1}: Loss = {self.loss(X, Y):.4f}")


class QTripleEncoder:
    """嵌入三元组为向量，并提取前 n 维用于量子分类器"""
    def __init__(self, n_dims=4):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.n_dims = n_dims

    def encode(self, s: str, p: str, o: str) -> List[float]:
        vec_s = self.model.encode(s)
        vec_p = self.model.encode(p)
        vec_o = self.model.encode(o)
        full_vec = (vec_s + vec_p + vec_o) / 3
        return full_vec[:self.n_dims].tolist()


if __name__ == '__main__':
    # 训练样本
    triples = [
        ("人类", "是", "生物"),
        ("爱因斯坦", "研究", "相对论")
    ]
    labels = [0, 1]  # 0=ontology, 1=instance

    encoder = QTripleEncoder()
    X_train = [encoder.encode(*t) for t in triples]

    qnn = QuantumClassifier()
    qnn.train(X_train, labels, epochs=50, lr=0.3)

    test_triple = ("苹果", "属于", "水果")
    test_vec = encoder.encode(*test_triple)
    pred = qnn.predict(test_vec)
    print(f"🔍 三元组分类结果：{pred} ({'ontology' if pred == 0 else 'instance'})")
