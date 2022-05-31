import torch
import torch.nn as nn


torch.manual_seed(0)


class Model(nn.Module):
    def __init__(self, in_features, intermediate_features, out_features):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(in_features, intermediate_features)
        self.linear2 = nn.Linear(intermediate_features, out_features)
        self.GELU = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        nn.init.normal_(self.linear1.weight, 0.0, 1.0)  # 平均0, 標準偏差1の正規分布による重みの初期化
        nn.init.normal_(self.linear2.weight, 0.0, 1.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.GELU(x)
        x = self.dropout(x)
        predicted_layer_num = self.linear2(x)

        return predicted_layer_num
