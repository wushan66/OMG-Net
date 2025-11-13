from typing import List

import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from loss import OrthogonalProjectionLoss

'''
Loss function for Matryoshka Representation Learning 
'''

class Matryoshka_CE_Loss(nn.Module):
    def __init__(self, relative_importance: List[float] = None, op_lambda=0.5, **kwargs):
        super(Matryoshka_CE_Loss, self).__init__()
        self.criterion_ce = nn.CrossEntropyLoss(**kwargs)
        self.criterion_op = OrthogonalProjectionLoss(0.3)
        self.relative_importance = relative_importance
        self.op_lambda = op_lambda

    def forward(self, output1, output2, target):
        # 1. 校验输出粒度数量一致
        assert len(output1) == len(output2), \
            f"output1（{len(output1)}个粒度）与output2（{len(output2)}个粒度）数量必须一致"

        # 2. 计算基础损失，并确保在同一设备（以output1的设备为准）
        device = output1[0].device  # 获取主分支设备
        ce_losses = torch.stack([self.criterion_ce(out.to(device), target.to(device)) for out in output1])

        # 强制辅助分支损失也在同一设备
        op_losses = torch.stack([self.criterion_op(out.to(device), target.to(device)) for out in output2])
        op_losses = op_losses.to(device)  # 显式移动到主设备

        # 3. 处理共享权重（确保与损失在同一设备）
        if self.relative_importance is None:
            shared_weights = torch.ones_like(ce_losses, device=device)
        else:
            assert len(self.relative_importance) == len(output1), \
                f"relative_importance长度（{len(self.relative_importance)}）与输出粒度数量（{len(output1)}）不匹配"
            shared_weights = torch.tensor(
                self.relative_importance,
                device=device,  # 明确指定设备
                dtype=ce_losses.dtype
            )

        # 4. 计算加权损失（所有张量已在同一设备）
        weighted_ce_loss = (shared_weights * ce_losses).sum()
        weighted_op_loss = (shared_weights * op_losses).sum()

        # 5. 融合损失
        total_loss = weighted_ce_loss + self.op_lambda * weighted_op_loss

        return total_loss, weighted_ce_loss, weighted_op_loss



class MRL_Linear_Layer(nn.Module):
    def __init__(self, nesting_list: List, num_classes=1000, efficient=False, **kwargs):
        super(MRL_Linear_Layer, self).__init__()
        self.nesting_list = nesting_list  # 嵌套维度列表，如[64, 128, 256, 512]
        self.num_classes = num_classes  # 分类任务类别数
        self.efficient = efficient  # 是否使用高效模式（共享权重）
        if self.efficient:
            # 高效模式：仅创建最大维度的线性层，其他维度共享其前缀权重
            setattr(self, f"nesting_classifier_{0}", nn.Linear(nesting_list[-1], self.num_classes, **kwargs))
        else:
            # 普通模式：为每个嵌套维度创建独立线性层
            for i, num_feat in enumerate(self.nesting_list):
                setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))

    def reset_parameters(self):
        # 重置线性层参数
        if self.efficient:
            self.nesting_classifier_0.reset_parameters()
        else:
            for i in range(len(self.nesting_list)):
                getattr(self, f"nesting_classifier_{i}").reset_parameters()

    def forward(self, x):
        nesting_logits = ()  # 存储经过线性层的输出（原逻辑）
        truncated_features = ()  # 新增：存储截断后未经过线性层的特征

        for i, num_feat in enumerate(self.nesting_list):
            # 1. 获取截断后未经过线性层的特征（核心修改）
            truncated_feat = x[:, :num_feat]  # 截取前num_feat维特征
            truncated_features += (truncated_feat,)  # 加入新元组

            # 2. 计算经过线性层的logits（保留原逻辑）
            if self.efficient:
                # 高效模式：共享最大维度线性层的前缀权重
                if self.nesting_classifier_0.bias is None:
                    logit = torch.matmul(
                        truncated_feat,  # 直接使用上面截取的特征
                        self.nesting_classifier_0.weight[:, :num_feat].t()
                    )
                else:
                    logit = torch.matmul(
                        truncated_feat,  # 直接使用上面截取的特征
                        self.nesting_classifier_0.weight[:, :num_feat].t()
                    ) + self.nesting_classifier_0.bias
                nesting_logits += (logit,)
            else:
                # 普通模式：使用对应维度的独立线性层
                logit = getattr(self, f"nesting_classifier_{i}")(truncated_feat)  # 直接使用上面截取的特征
                nesting_logits += (logit,)

        # 返回两个元组：(经过线性层的logits, 截断后未经过线性层的特征)
        return nesting_logits, truncated_features


class FixedFeatureLayer(nn.Linear):
    '''
    For our fixed feature baseline, we just replace the classification layer with the following.
    It effectively just look at the first "in_features" for the classification.
    '''

    def __init__(self, in_features, out_features, **kwargs):
        super(FixedFeatureLayer, self).__init__(in_features, out_features, **kwargs)

    def forward(self, x):
        if not (self.bias is None):
            out = torch.matmul(x[:, :self.in_features], self.weight.t()) + self.bias
        else:
            out = torch.matmul(x[:, :self.in_features], self.weight.t())
        return out
