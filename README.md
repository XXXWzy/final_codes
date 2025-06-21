# NeRF与3D Gaussian Splatting物体重建实验

本项目实现了基于神经辐射场(NeRF)和3D高斯点绘制(3D Gaussian Splatting)的物体重建和新视图合成，包含完整的实验流水线和详细的对比分析。

## 🎯 项目概述

### 实验目标
- **任务1**: 使用TensoRF(NeRF加速技术)实现物体重建和新视图合成
- **任务2**: 使用3D Gaussian Splatting实现物体重建和新视图合成
- **对比分析**: 比较原版NeRF、TensoRF和3D GS三种方法的性能

### 主要特性
- ✅ 完整的数据处理流水线(COLMAP SfM重建)
- ✅ 自动图像质量评估和选择
- ✅ TensoRF模型训练和渲染
- ✅ 360度新视角视频生成
- ✅ 定量评价指标(PSNR、SSIM、LPIPS)
- 🔄 3D Gaussian Splatting实现(框架已准备)

## 📁 项目结构

```
task03/
├── scripts/                          # 主要脚本
│   ├── prepare_data.py               # 标准数据预处理
│   ├── prepare_data_relaxed.py       # 优化的数据预处理
│   ├── train_tensorf.py              # TensoRF训练脚本
│   └── render_tensorf.py             # 渲染和视频生成
├── models/                           # 模型实现
│   ├── nerf/tensorf.py              # TensoRF模型
│   └── utils/                       
│       ├── dataset.py               # 数据集工具
│       ├── metrics.py               # 评价指标
│       └── rendering.py             # 渲染工具
├── configs/                          # 配置文件
│   └── tensorf_small.yaml          # TensoRF训练配置
├── data/                            # 数据目录
│   ├── raw_images/                  # 原始图像(用户提供)
│   ├── colmap/                      # COLMAP重建结果
│   └── colmap_relaxed/              # 优化重建结果
├── experiments/                      # 实验结果
│   └── tensorf_experiment_05/       # TensoRF训练结果
├── renders/                         # 渲染输出
│   ├── tensorf_360/                 # 测试视频
│   └── tensorf_360_full/            # 完整360度视频
├── TASK1_NERF_REPORT.md             # 任务1详细报告
├── TASK2_3DGS_REPORT.md             # 任务2详细报告
└── PROJECT_SUMMARY.md               # 项目技术总结
```

## 🚀 快速开始

### 环境准备
```bash
# 创建conda环境
conda create -n nerf-3dgs python=3.9
conda activate nerf-3dgs

# 安装依赖(macOS)
pip install -r requirements_macos.txt

# 安装COLMAP(macOS)
brew install colmap
```

### 数据准备
1. **图像采集**: 围绕物体拍摄100+张多角度图像
2. **放置数据**: 将图像放入`data/raw_images/`目录
3. **数据处理**: 
```bash
# 使用优化的数据预处理
python scripts/prepare_data_relaxed.py \
    --input_dir data/raw_images \
    --output_dir data/colmap_relaxed \
    --max_images 30
```

### TensoRF训练
```bash
python scripts/train_tensorf.py \
    --config configs/tensorf_small.yaml \
    --data_dir data/colmap_relaxed \
    --exp_name tensorf_experiment \
    --device cpu
```

### 新视角渲染
```bash
# 生成360度视频
python scripts/render_tensorf.py \
    --checkpoint experiments/tensorf_experiment/checkpoints/final_model.pth \
    --config experiments/tensorf_experiment/config.yaml \
    --data_dir data/colmap_relaxed \
    --output_dir renders/360_video \
    --n_frames 120 \
    --radius 3.0
```

## 📊 实验结果

### 已完成实验(任务1 - TensoRF)
- **训练时长**: 36,288秒 (~10小时)
- **训练PSNR**: 20.51 dB
- **验证PSNR**: 22.67 dB
- **模型参数**: 3,201,939个
- **渲染结果**: 360度高质量视频

### 关键文件位置
- 训练好的模型: `experiments/tensorf_experiment_05/checkpoints/final_model.pth`
- 360度视频: `renders/tensorf_360/novel_view_360.mp4`
- 测试渲染: `renders/tensorf_360/test_views/`

## 📖 详细报告

### [任务1报告: 基于NeRF的物体重建](TASK1_NERF_REPORT.md)
完整的TensoRF实验报告，包含：
- 详细的技术原理和对比分析
- 完整的实验设置和训练过程
- 定量和定性结果分析
- 训练曲线和性能指标

### [任务2报告: 基于3D Gaussian Splatting的物体重建](TASK2_3DGS_REPORT.md)
3D Gaussian Splatting实验框架，包含：
- 3D GS技术原理详解
- 实验设置和预期结果
- 与NeRF方法的全面对比
- 性能分析和应用建议

### [项目技术总结](PROJECT_SUMMARY.md)
技术实现的详细文档，包含：
- 关键技术挑战和解决方案
- 代码架构和模块说明
- 性能优化和设备适配

## 🔧 技术亮点

### 数据预处理优化
- **问题**: COLMAP重建成功率低(2/103张)
- **解决**: 
  - 支持SIMPLE_RADIAL相机模型
  - 基于清晰度的图像自动选择
  - 放宽匹配参数提高成功率(13/30张)

### 模型兼容性
- **MPS设备适配**: 解决Apple Silicon Mac的数据类型问题
- **批处理优化**: 动态调整批次大小避免内存溢出
- **设备自动检测**: 支持CPU/MPS/CUDA自动切换

### 渲染优化
- **分块渲染**: 支持大场景的内存友好渲染
- **360度路径**: 自动生成平滑的环绕相机轨迹
- **高质量输出**: 生成MP4格式的高质量视频

## 📈 性能对比

| 方法 | 训练时间 | 渲染速度 | PSNR | 模型大小 | 适用场景 |
|------|----------|----------|------|----------|----------|
| **原版NeRF** | >24小时 | <1 FPS | ~22 dB | ~5MB | 学术研究 |
| **TensoRF** | ~10小时 | ~1 FPS | 22.67 dB | ~12MB | 快速原型 |
| **3D GS** | ~2小时 | >30 FPS | >25 dB | ~50MB | 实时应用 |

## 🛠️ 依赖要求

### 核心依赖
- Python 3.9+
- PyTorch 1.13+
- NumPy, OpenCV
- imageio, tqdm
- COLMAP 3.7+

### 硬件要求
- **最低**: 8GB RAM, Apple Silicon Mac
- **推荐**: 16GB+ RAM, NVIDIA GPU (CUDA)
- **存储**: 至少5GB可用空间

## 🐛 常见问题

### COLMAP重建失败
```bash
# 检查图像质量
python scripts/prepare_data_relaxed.py --input_dir data/raw_images --analyze_only

# 使用更宽松的参数
python scripts/prepare_data_relaxed.py --input_dir data/raw_images --output_dir data/colmap_relaxed
```

### MPS设备错误
```bash
# 强制使用CPU
python scripts/train_tensorf.py --device cpu

# 检查PyTorch MPS支持
python -c "import torch; print(torch.backends.mps.is_available())"
```

### 内存不足
```bash
# 降低图像分辨率
python scripts/train_tensorf.py --img_downscale 4

# 减少批次大小
# 编辑configs/tensorf_small.yaml中的batch_size
```

## 📝 引用和参考

### 相关论文
- **TensoRF**: Tensorial Radiance Fields (ECCV 2022)
- **3D Gaussian Splatting**: 3D Gaussian Splatting for Real-Time Radiance Field Rendering (SIGGRAPH 2023)
- **NeRF**: Neural Radiance Fields (ECCV 2020)

### 代码参考
- [TensoRF官方实现](https://github.com/apchenstu/TensoRF)
- [3D Gaussian Splatting官方](https://github.com/graphdeco-inria/gaussian-splatting)

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境
```bash
git clone https://github.com/your-username/nerf-3dgs-experiment
cd nerf-3dgs-experiment
conda env create -f environment.yml
```

### 提交格式
- `feat:` 新功能
- `fix:` 错误修复
- `docs:` 文档更新
- `perf:` 性能优化

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 👥 作者

- **主要开发者**: [您的姓名]
- **联系方式**: [您的邮箱]
- **GitHub**: [您的GitHub用户名]

---

**最后更新**: 2024年6月  
**项目状态**: 任务1已完成，任务2框架已准备 