from typing import Tuple, List, Optional
import time
from datetime import datetime  # 添加datetime模块

class DeepxIR:
    def __init__(self, 
                name:str,
                dtype:str,
                args: List[str], 
                returns: List[str], 
                grad: bool = False,
                args_grad: Optional[List[str]] = None,
                returns_grad: Optional[List[str]] = None):
        """
        初始化操作节点
        Args:
            args: 输入参数名称列表（如["input", "weight"]）
            returns: 输出参数名称列表（如["output"]）
            grad: 是否需要进行梯度计算
            args_grad: 输入参数的梯度名称列表（与args一一对应，空字符串表示无梯度）
            returns_grad: 输出参数的梯度名称列表（与returns一一对应）
        """
        # 基础参数校验
        if grad:
            if args_grad is None:
                args_grad = [""] * len(args)
            if returns_grad is None:
                returns_grad = [""] * len(returns)
                
            if len(args_grad) != len(args):
                raise ValueError("args_grad必须与args长度一致")
            if len(returns_grad) != len(returns):
                raise ValueError("returns_grad必须与returns长度一致")

        self._name = name  
        self._dtype = dtype
        self._args = args
        self._returns = returns
        self._grad = grad
        self._args_grad = args_grad if grad else []
        self._returns_grad = returns_grad if grad else []
        self._id=None
        self._created_at=time.time()
        self._sent_at=None

    def __str__(self):
        if self._dtype == None or self._dtype == '':
            parts = [self._name]
        else:
            parts = [f"{self._name}@{self._dtype}"]  # 常规类型显示
        
        # 处理输入参数
        for i in range(len(self._args)):
            arg_part = str(self._args[i])
            if self._grad and self._args_grad[i]:
                arg_part += f"({self._args_grad[i]})"
            parts.append(arg_part)
        
        # 添加箭头
        arrow = "->" if not self._grad else "<-"
        parts.append(arrow)
        
        # 处理输出参数
        for i in range(len(self._returns)):
            ret_part = str(self._returns[i])
            if self._grad and self._returns_grad[i]:
                ret_part += f"({self._returns_grad[i]})"
            parts.append(ret_part)

        parts.append("//")
        parts.append(f"id={self._id}")
        parts.append(f"created_at={self._created_at}")
        parts.append(f"sent_at={self._sent_at}")
        return ' '.join(parts)

class DeepxIRResp:
    #'1 ok examplemsg // recv_at=1741494459006 start_at=1741494459006 finish_at=1741494459006'
    def __init__(self,s:str):
        self._id=None
        self._result=""
        self._message=''
        #extra info
        self._recv_at=None  
        self._start_at=None
        self._finish_at=None
        
        # 解析响应字符串
        if s and isinstance(s, str):
            # 首先按 "//" 分割为前后两部分
            parts = s.split("//", 1)
            
            if len(parts) >= 1:
                # 处理前半部分 ID、结果和消息
                front_parts = parts[0].strip().split(" ", 2)
                
                if len(front_parts) >= 1:
                    self._id = front_parts[0]
                
                if len(front_parts) >= 2:
                    self._result = front_parts[1]
                
                if len(front_parts) >= 3:
                    self._message = front_parts[2]
            
            # 处理后半部分的时间戳信息
            if len(parts) >= 2:
                extra_info = parts[1].strip()
                extra_parts = extra_info.split()
                
                for part in extra_parts:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        if key == "recv_at":
                            # 将毫秒时间戳转换为datetime对象
                            self._recv_at = datetime.fromtimestamp( float(value) / 1000.0)
                        elif key == "start_at":
                            self._start_at =datetime.fromtimestamp( float(value) / 1000.0)
                        elif key == "finish_at":
                            self._finish_at = datetime.fromtimestamp( float(value) / 1000.0)
 
    def __str__(self) -> str:
        parts=[]
        parts.append(self._id)
        parts.append(self._result)
        parts.append(self._message)
        parts.append("//")
        if self._recv_at is not None:
            parts.append(f"recv_at={self._recv_at.strftime('%H:%M:%S.%f')[:-3]}")
        if self._start_at is not None:
            parts.append(f"start_at={self._start_at.strftime('%H:%M:%S.%f')[:-3]}")
        if self._finish_at is not None:
            parts.append(f"finish_at={self._finish_at.strftime('%H:%M:%S.%f')[:-3]}")
        return ' '.join(parts)