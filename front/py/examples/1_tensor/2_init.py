import sys
sys.path.append('/home/lipeng/code/git.array2d.com/ai/deepx/front/py')  # 将项目根目录添加到Python路径

from deepx.tensor import zeros, ones, arange

def printall(t):
   print("t=",t)
   print("t.shape=",t.shape)
   print("t.shape[0]=",t.shape[0])
   print("t.stride=",t.stride)
   print("t.stride[0]=",t.stride[0])
   print("t.dim=",t.dim)
   print("t.ndimension=",t.ndimension)
   print("t.numel=",t.numel)
   print("t.dtype=", t.dtype)
if __name__ == "__main__":
   args = sys.argv[1:]
   caseid = 0
   if len(args) > 0:
      caseid = int(args[0])

   if caseid == 0:
      t=zeros(1)
      printall(t)
   elif caseid == 1:
      t=ones()
      printall(t)
   elif caseid == 2:
      t=arange()
      printall(t)
   elif caseid == 3:
      t=arange(1,10,2)
      printall(t)
   elif caseid == 4:
      t=arange(1,10,2)
      printall(t)
