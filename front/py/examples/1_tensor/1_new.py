import sys
sys.path.append('/home/lipeng/code/git.array2d.com/ai/deepx/front/py')  # 将项目根目录添加到Python路径

from deepx.tensor import Tensor


def main():
   t = Tensor([1,2,3])
   print(t)
   print(t.shape)
   print(t.shape[0])
   print(t.stride)
   print(t.stride[0])
   print(t.dim)
   print(t.ndimension)
   print(t.numel)
   print(t.dtype)

if __name__ == "__main__":
    main() 