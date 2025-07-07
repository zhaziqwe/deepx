#!/bin/bash

echo "🚀 DeepX框架安装脚本"
echo "===================="

# 检查是否在正确的目录
if [ ! -f "setup.py" ]; then
    echo "❌ 错误：请在front/py目录下运行此脚本"
    exit 1
fi

# 检查uv是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ 错误：未找到uv，请先安装uv"
    echo "   安装命令：curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 创建新的虚拟环境
echo "🔧 创建新的虚拟环境..."
uv venv .venv

# 激活虚拟环境
echo "📦 激活虚拟环境..."
source .venv/bin/activate

# 升级pip
echo "⬆️  升级pip..."
uv pip install --upgrade pip

# 安装依赖
echo "📦 安装依赖包..."
uv pip install -r requirements.txt

# 安装deepx包（开发模式）
echo "🔧 安装deepx包（开发模式）..."
uv pip install -e .

# 验证安装
echo "✅ 验证安装..."
python -c "import deepx; print('✅ deepx包安装成功')"
python -c "import deepxutil; print('✅ deepxutil包安装成功')"

echo ""
echo "🎉 安装完成！"
echo "===================="
echo "🎯 现在您可以运行examples中的程序："
echo "   cd examples/1_tensor"
echo "   python 1_new.py"
echo ""
echo "📚 更多示例请查看examples目录"
echo ""
echo "💡 提示：每次使用前请激活虚拟环境："
echo "   source .venv/bin/activate" 