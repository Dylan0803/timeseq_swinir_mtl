import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMSELoss(nn.Module):
    """
    加权MSE损失函数
    
    对背景区域(低于阈值的区域)和有浓度区域(高于阈值的区域)使用不同权重，
    用来解决对于没有浓度区域过分关注的问题。
    
    参数:
        alpha (float): 有浓度区域(高值区域)的权重
        beta (float): 背景区域(低值区域)的权重
        threshold (float): 区分背景和有浓度区域的阈值
    """
    def __init__(self, alpha=1.0, beta=0.3, threshold=0.05):
        super(WeightedMSELoss, self).__init__()
        self.alpha = alpha  # 有浓度区域的权重
        self.beta = beta    # 背景区域的权重
        self.threshold = threshold  # 区分背景和有浓度区域的阈值
        
    def forward(self, pred, target):
        """
        计算加权MSE损失
        
        参数:
            pred (torch.Tensor): 模型预测的输出
            target (torch.Tensor): 目标真实值
            
        返回:
            torch.Tensor: 加权MSE损失值
        """
        # 创建掩码区分背景和有浓度区域
        high_density_mask = (target > self.threshold)
        low_density_mask = ~high_density_mask
        
        # 计算两个区域的MSE损失
        mse_high = F.mse_loss(
            pred * high_density_mask.float(), 
            target * high_density_mask.float(), 
            reduction='sum'
        )
        
        mse_low = F.mse_loss(
            pred * low_density_mask.float(), 
            target * low_density_mask.float(), 
            reduction='sum'
        )
        
        # 计算像素数量用于归一化
        num_high_pixels = torch.sum(high_density_mask.float())
        num_low_pixels = torch.sum(low_density_mask.float())
        
        # 避免除零错误
        num_high_pixels = torch.max(num_high_pixels, torch.tensor(1.0).to(pred.device))
        num_low_pixels = torch.max(num_low_pixels, torch.tensor(1.0).to(pred.device))
        
        # 计算加权损失
        loss = (self.alpha * mse_high / num_high_pixels) + (self.beta * mse_low / num_low_pixels)
        
        return loss 