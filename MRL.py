from typing import List

import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from loss import OrthogonalProjectionLoss

class Matryoshka_CE_Loss(nn.Module):
    def __init__(self, relative_importance: List[float] = None, op_lambda=0, **kwargs):
        super(Matryoshka_CE_Loss, self).__init__()
        self.criterion_ce = nn.CrossEntropyLoss(**kwargs)
        self.criterion_op = OrthogonalProjectionLoss(0)
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

        return total_loss,weighted_ce_loss,weighted_op_loss
   
