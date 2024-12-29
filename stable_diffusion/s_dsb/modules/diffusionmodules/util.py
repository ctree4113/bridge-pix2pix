import importlib
import torch
import torch.nn as nn
import numpy as np
from collections import abc
from einops import rearrange
from functools import partial
import multiprocessing as mp
from threading import Thread
from queue import Queue
from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Any, Callable, List, Union

def log_txt_as_img(wh: tuple, xc: List[str], size: int = 10) -> torch.Tensor:
    """将文本转换为图像用于日志记录
    Args:
        wh: (width, height)元组
        xc: 要绘制的文本列表
        size: 字体大小
    Returns:
        文本图像张量
    """
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts

def _do_parallel_data_prefetch(func: Callable, Q: Union[mp.Queue, Queue], 
                             data: Any, idx: int, idx_to_fn: bool = False) -> None:
    """并行数据预取的工作函数"""
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")

def parallel_data_prefetch(
    func: Callable,
    data: Union[np.ndarray, List],
    n_proc: int,
    target_data_type: str = "ndarray",
    cpu_intensive: bool = True,
    use_worker_id: bool = False
) -> Union[np.ndarray, List]:
    """并行数据预取
    Args:
        func: 要并行执行的函数
        data: 输入数据
        n_proc: 进程数
        target_data_type: 目标数据类型
        cpu_intensive: 是否CPU密集型任务
        use_worker_id: 是否使用worker ID
    Returns:
        处理后的数据
    """
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print('WARNING: "data" is a dict: Using only its values and disregarding keys.')
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"Data must be ndarray or Iterable, got {type(data)}."
        )

    # 选择进程或线程
    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread

    # 准备参数
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(
                [data[i: i + step] for i in range(0, len(data), step)]
            )
        ]

    # 启动进程
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes.append(p)

    # 收集结果
    print(f"Start prefetching...")
    import time
    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()
        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    # 整理输出
    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == 'list':
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res

def exists(x: Any) -> bool:
    """检查变量是否存在"""
    return x is not None

def default(val: Any, d: Any) -> Any:
    """返回默认值"""
    if exists(val):
        return val
    return d() if isfunction(d) else d

def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """计算张量在非batch维度上的平均值"""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def count_params(model: nn.Module, verbose: bool = False) -> int:
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params

def instantiate_from_config(config: dict) -> Any:
    """从配置实例化对象"""
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    
    # 获取目标类
    target = get_obj_from_str(config["target"])
    
    # 获取参数
    params = config.get("params", {})
    
    # 实例化
    try:
        return target(**params)
    except Exception as e:
        print(f"Error instantiating {target} with params {params}")
        raise e

def get_obj_from_str(string: str, reload: bool = False) -> Any:
    """从字符串获取对象"""
    try:
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)
    except Exception as e:
        print(f"Error loading object {string}")
        raise e

# S-DSB特有工具函数
def get_gamma_schedule(
    schedule_type: str,
    num_timesteps: int,
    gamma_min: float = 1e-4,
    gamma_max: float = 0.1
) -> torch.Tensor:
    """获取gamma schedule
    
    Args:
        schedule_type: 调度类型 ("linear" or "cosine")
        num_timesteps: 时间步数
        gamma_min: 最小gamma值
        gamma_max: 最大gamma值
    """
    if schedule_type == "linear":
        gammas = torch.linspace(gamma_min, gamma_max, num_timesteps // 2)
        return torch.cat([gammas, gammas.flip(dims=(0,))])
    elif schedule_type == "cosine":
        steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alpha_bar = torch.cos(((steps / num_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
        gammas = torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], gamma_min, gamma_max)
        return gammas
    else:
        raise ValueError(f"Unknown schedule type {schedule_type}")

def reparameterize_flow(
    x: torch.Tensor,
    t: torch.Tensor,
    gamma_t: torch.Tensor,
    direction: str = "forward"
) -> torch.Tensor:
    """Flow重参数化
    
    Args:
        x: 输入张量
        t: 时间步
        gamma_t: gamma值
        direction: 方向 ("forward" or "backward")
    """
    if direction == "forward":
        return (x - x.mean()) / gamma_t.sqrt()
    else:
        return (x - x.mean()) / (1 - gamma_t).sqrt()

def reparameterize_term(
    x: torch.Tensor,
    x_start: torch.Tensor,
    t: torch.Tensor,
    gamma_t: torch.Tensor,
    direction: str = "forward"
) -> torch.Tensor:
    """Terminal重参数化
    
    Args:
        x: 输入张量
        x_start: 起始张量
        t: 时间步
        gamma_t: gamma值
        direction: 方向 ("forward" or "backward")
    """
    if direction == "forward":
        return torch.cat([x, x_start], dim=1)
    else:
        return torch.cat([x, x_start], dim=1)

def extract(
    a: torch.Tensor,
    t: torch.Tensor,
    x_shape: tuple
) -> torch.Tensor:
    """从batch中提取指定时间步的值"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def noise_like(
    shape: tuple,
    device: torch.device,
    repeat: bool = False
) -> torch.Tensor:
    """生成噪声张量"""
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()