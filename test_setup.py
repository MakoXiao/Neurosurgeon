"""快速测试脚本 - 验证所有模块是否正确安装和配置"""
import sys
import os

print("="*60)
print("测试实验环境配置")
print("="*60)

# 测试Python版本
print(f"\n1. Python版本: {sys.version.split()[0]}")
print(f"   Python路径: {sys.executable}")

# 测试PyTorch
try:
    import torch
    print(f"\n2. PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"\n2. ✗ PyTorch导入失败: {e}")
    sys.exit(1)

# 测试其他依赖
print("\n3. 检查依赖:")
for dep in ['numpy', 'matplotlib', 'tqdm', 'torchvision']:
    try:
        __import__(dep)
        print(f"   ✓ {dep}")
    except ImportError:
        print(f"   ✗ {dep}")

# 测试数据集
print("\n4. 检查数据集:")
data_dir = "/opt/03-ai/01-proj/Neurosurgeon/data/caltech-101"
if os.path.exists(data_dir):
    print(f"   ✓ 数据集目录存在")
    categories_dir = os.path.join(data_dir, "101_ObjectCategories")
    if os.path.exists(categories_dir):
        categories = [d for d in os.listdir(categories_dir) if os.path.isdir(os.path.join(categories_dir, d))]
        print(f"   ✓ 找到 {len(categories)} 个类别")
else:
    print(f"   ✗ 数据集目录不存在")

# 测试项目模块
print("\n5. 测试项目模块:")
modules = [
    ('dataset.caltech101_loader', '数据加载器'),
    ('models.model_zoo', '模型定义'),
    ('compression.pruning_compression', '剪枝压缩'),
    ('rl_agent.hybrid_ppo', 'PPO算法'),
    ('rl_agent.state_reward', '状态和奖励'),
    ('environment.collaborative_env', '协同推理环境'),
]

all_ok = True
for mod, desc in modules:
    try:
        __import__(mod)
        print(f"   ✓ {desc} ({mod})")
    except Exception as e:
        print(f"   ✗ {desc} ({mod}): {str(e)[:60]}")
        all_ok = False

# 快速功能测试
print("\n6. 快速功能测试:")
try:
    from models.model_zoo import get_model
    import torch
    model = get_model('resnet18', num_classes=101, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    print(f"   ✓ 模型创建和推理正常 (输出形状: {output.shape})")
except Exception as e:
    print(f"   ✗ 模型测试失败: {e}")
    all_ok = False

print("\n" + "="*60)
if all_ok:
    print("✓ 环境配置正确！可以开始实验。")
    print("\n快速开始:")
    print("  bash run_experiments.sh")
    print("\n或分步运行:")
    print("  python train_models.py --model resnet18 --epochs 10 --device cpu")
else:
    print("✗ 部分模块导入失败，请检查错误。")
print("="*60)
