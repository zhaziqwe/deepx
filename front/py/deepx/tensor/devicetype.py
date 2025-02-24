from enum import Enum

class DeviceType(Enum):
    CPU = 'cpu'
    CUDA = 'cuda'
class Device:
    def __init__(self, type: DeviceType, device_index=0):
        self.type = type
        self.device_index = device_index  # 仅对CUDA有效
        
    @classmethod
    def from_string(cls, device_str):
        """解析字符串格式：'cuda:0' 或 'cpu'"""
        if ':' in device_str:
            type_str, index_str = device_str.split(':')
            return cls(DeviceType(type_str), int(index_str))
        return cls(DeviceType(device_str))
    
    def __eq__(self, other):
        return self.type == other.type and self.device_index == other.device_index
    
    def __repr__(self):
        if self.type == DeviceType.CPU:
            return 'Device(type=cpu)'
        return f'Device(type={self.type.value}, index={self.device_index})'
    
Device.CPU = Device(DeviceType.CPU)
Device.CUDA = Device(DeviceType.CUDA)