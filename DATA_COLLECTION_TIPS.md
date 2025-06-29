# 数据收集指南 - NeRF & 3D Gaussian Splatting

## 当前问题分析

您的COLMAP重建只成功重建了2张图片（共103张），这是一个常见问题。主要原因可能包括：

1. **重叠度不足**: 相邻图片之间需要60-80%的重叠
2. **图片质量问题**: 模糊、过曝、欠曝的图片
3. **相机运动过大**: 帧间变化过大导致特征匹配失败
4. **纹理不足**: 物体表面缺乏足够的纹理特征

## 改进建议

### 1. 重新拍摄数据

#### 相机设置
```bash
# 推荐设置
- 分辨率: 1080p 或更高
- 格式: JPG (高质量) 或 RAW
- 焦距: 固定焦距，避免变焦
- 对焦: 手动对焦，锁定焦距
- 曝光: 手动曝光，保持一致
- ISO: 尽量低 (100-400)
- 快门速度: 1/125s 或更快（避免模糊）
```

#### 拍摄技巧
```bash
# 运动模式
1. 围绕物体拍摄:
   - 保持与物体的距离一致
   - 每5-10度拍一张
   - 总共需要36-72张图片

2. 多层拍摄:
   - 至少3个高度层次
   - 每层都围绕物体一圈
   - 包含俯视和仰视角度

3. 重叠要求:
   - 相邻图片重叠60-80%
   - 避免跳跃式拍摄
```

### 2. 使用现有数据的解决方案

如果无法重新拍摄，我们可以尝试以下方法：

#### 方法1: 手动选择最佳图片
```bash
# 从103张图片中手动选择质量最好的20-30张
# 确保它们有良好的重叠度和清晰度
mkdir data/selected_images
# 手动复制选择的图片到这个目录
```

#### 方法2: 使用更宽松的COLMAP参数
```bash
# 修改COLMAP参数以提高匹配成功率
colmap feature_extractor \
    --database_path database.db \
    --image_path images \
    --ImageReader.single_camera 1 \
    --SiftExtraction.max_image_size 3200 \
    --SiftExtraction.max_num_features 8192
```

#### 方法3: 使用顺序匹配
```bash
# 使用顺序匹配而不是穷举匹配
colmap sequential_matcher \
    --database_path database.db \
    --SequentialMatching.overlap 10
```

## 数据质量检查清单

### 图片质量检查
- [ ] 所有图片都清晰无模糊
- [ ] 曝光均匀，无过曝或欠曝
- [ ] 物体在所有图片中都完整可见
- [ ] 背景相对简单且一致

### 相机运动检查
- [ ] 相邻图片间的视角变化较小
- [ ] 覆盖物体的所有角度
- [ ] 包含多个高度层次
- [ ] 有足够的重叠区域

### 场景设置检查
- [ ] 光照条件稳定
- [ ] 物体有丰富的纹理特征
- [ ] 背景不会干扰特征检测
- [ ] 物体表面非反光

## 实验继续方案

### 选项1: 使用现有的2张图片进行概念验证
```bash
# 虽然只有2张图片，但可以用来测试管道
# 创建一个最小的训练集
python scripts/create_minimal_dataset.py
```

### 选项2: 下载示例数据集
```bash
# 下载标准的NeRF测试数据集
wget https://drive.google.com/file/d/1JDT-in8wUP-1YfE3mQX_-fD8wPd7XGDL/view?usp=sharing
# 使用标准数据集进行实验
```

### 选项3: 生成合成数据
```bash
# 使用Blender生成合成数据
# 这样可以确保完美的相机参数和充足的视角
```

## 快速修复当前数据

让我们尝试修改COLMAP参数来提高重建成功率：

```bash
# 1. 清理现有结果
rm -rf data/colmap/sparse/*

# 2. 使用更宽松的参数重新运行
python scripts/prepare_data_relaxed.py --input_dir data/raw_images --output_dir data/colmap
```

## 质量评估指标

### COLMAP重建质量
- 重建图片数量 / 总图片数量 > 80%
- 3D点云密度合理
- 相机位置分布均匀

### NeRF训练准备
- 训练集 > 50张图片
- 测试集 > 10张图片
- 相机参数准确

## 下一步行动

1. **立即行动**: 使用现有的2张图片测试训练管道
2. **短期计划**: 手动选择更多高质量图片重新运行COLMAP
3. **长期计划**: 重新拍摄数据或使用标准数据集

## 学习要点

这个问题很好地展示了：
- 数据质量对3D重建的重要性
- COLMAP的局限性和参数调优
- 真实世界数据收集的挑战
- 实验设计的迭代性质

记录这些经验对最终报告很有价值！ 