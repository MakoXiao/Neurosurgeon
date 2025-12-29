"""
环境检查脚本
检查所有依赖包和配置是否正确
"""
import sys
import os

def check_python_version():
    """检查Python版本"""
    print(f"Python版本: {sys.version}")
    version_info = sys.version_info
    if version_info.major == 3 and version_info.minor >= 7:
        print("✅ Python版本符合要求")
        return True
    else:
        print("❌ Python版本过低，需要Python 3.7+")
        return False

def check_imports():
    """检查所有必需的包"""
    required_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'PIL',
        'matplotlib',
        'seaborn',
        'pandas',
        'tqdm',
        'psutil'
    ]
    
    print("\n检查依赖包:")
    all_ok = True
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安装")
            all_ok = False
    
    return all_ok

def check_cuda():
    """检查CUDA可用性"""
    try:
        import torch
        print(f"\n检查CUDA:")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"当前GPU: {torch.cuda.get_device_name(0)}")
            print("✅ CUDA配置正常")
            return True
        else:
            print("⚠️  CUDA不可用，将使用CPU")
            return True
    except Exception as e:
        print(f"❌ CUDA检查失败: {e}")
        return False

def check_data():
    """检查数据集"""
    print("\n检查数据集:")
    data_dir = '/opt/03-ai/01-proj/Neurosurgeon/data/caltech-101/101_ObjectCategories'
    
    if os.path.exists(data_dir):
        categories = [d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d))]
        print(f"✅ Caltech-101数据集存在")
        print(f"   类别数: {len(categories)}")
        return True
    else:
        print(f"❌ 数据集不存在: {data_dir}")
        return False

def check_project_structure():
    """检查项目结构"""
    print("\n检查项目结构:")
    
    required_dirs = [
        'models',
        'dataset', 
        'compression',
        'rl_agent',
        'environment',
        'experiments'
    ]
    
    required_files = [
        'train_models.py',
        'train_rl_agent.py',
        'run_experiments.sh'
    ]
    
    all_ok = True
    
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - 目录不存在")
            all_ok = False
    
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - 文件不存在")
            all_ok = False
    
    return all_ok

def check_modules():
    """检查模块导入"""
    print("\n检查模块导入:")
    
    modules_to_check = [
        ('dataset.caltech101_loader', 'get_caltech101_dataloaders'),
        ('models.model_zoo', 'get_model'),
        ('compression.pruning_compression', 'AdaptivePruningCompressor'),
        ('rl_agent.hybrid_ppo', 'HybridPPO'),
        ('rl_agent.state_reward', 'StateSpace'),
        ('rl_agent.state_reward', 'RewardFunction'),
        ('environment.collaborative_env', 'CollaborativeInferenceEnv'),
    ]
    
    all_ok = True
    
    for module_name, class_name in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name} - {str(e)}")
            all_ok = False
    
    return all_ok

def create_directories():
    """创建必要的目录"""
    print("\n创建必要的目录:")
    
    dirs_to_create = [
        'checkpoints',
        'rl_agents',
        'results',
        'figures'
    ]
    
    for dir_name in dirs_to_create:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ 创建目录: {dir_name}/")
        else:
            print(f"✅ 目录已存在: {dir_name}/")

def main():
    """主函数"""
    print("="*60)
    print("环境检查脚本")
    print("="*60)
    
    os.chdir('/opt/03-ai/01-proj/Neurosurgeon')
    
    # 检查各项配置
    checks = {
        'Python版本': check_python_version(),
        '依赖包': check_imports(),
        'CUDA': check_cuda(),
        '数据集': check_data(),
        '项目结构': check_project_structure(),
        '模块导入': check_modules()
    }
    
    # 创建必要的目录
    create_directories()
    
    # 输出总结
    print("\n" + "="*60)
    print("检查总结")
    print("="*60)
    
    for check_name, result in checks.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name}: {status}")
    
    all_passed = all(checks.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ 所有检查通过！环境配置完成！")
        print("\n下一步:")
        print("1. 运行快速测试: python train_models.py --model resnet18 --epochs 2")
        print("2. 运行完整实验: bash run_experiments.sh")
    else:
        print("❌ 部分检查失败，请根据上述提示修复问题")
    print("="*60 + "\n")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)


