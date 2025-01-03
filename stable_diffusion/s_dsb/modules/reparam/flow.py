import torch

def reparameterize_flow(x_0, t, gamma_t, direction="forward"):
    """流式重参数化
    
    Args:
        x_0: 初始状态
        t: 时间步
        gamma_t: gamma值
        direction: 方向,'forward' 或 'backward'
        
    Returns:
        重参数化后的状态
    """
    # 生成随机噪声
    noise = torch.randn_like(x_0)
    
    # 根据方向计算重参数化
    if direction == "forward":
        # 前向过程: x_t = sqrt(1-gamma_t)x_0 + sqrt(gamma_t)eps
        x_t = torch.sqrt(1 - gamma_t) * x_0 + torch.sqrt(gamma_t) * noise
    else:
        # 反向过程: x_t = sqrt(gamma_t)x_0 + sqrt(1-gamma_t)eps 
        x_t = torch.sqrt(gamma_t) * x_0 + torch.sqrt(1 - gamma_t) * noise
        
    return x_t

def reparameterize(
    x: torch.Tensor,
    t: torch.Tensor,
    gamma_t: torch.Tensor,
    mode: str = "flow",
    direction: str = "forward"
) -> torch.Tensor:
    """统一的重参数化接口
    
    Args:
        x: 输入张量
        t: 时间步
        gamma_t: gamma值
        mode: 重参数化模式 ("flow" or "terminal")
        direction: 方向 ("forward" or "backward")
    """
    if mode == "flow":
        return reparameterize_flow(x, t, gamma_t, direction)
    else:
        return reparameterize_term(x, t, gamma_t, direction)