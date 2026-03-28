# 分场景 CFG Weight Grid Search 实验指南

## 实验目的
测试不同场景类型（左转/右转/变道/跟车等）是否需要不同的 CFG 权重 w。

## 环境要求
- AutoDL 实例（RTX 4090，建议 ≥50GB 数据盘）
- 已安装 `flow_planner` conda 环境
- 已有 nuPlan 验证数据（val 数据集 + maps）
- 已有模型 checkpoint（epoch_120.pth）

## 快速开始

### 1. 拉取最新代码
```bash
cd /root/Flow-Planner
git pull origin main
```

### 2. 检查并修改配置
```bash
vim scripts/run_w_grid_search.sh
```
需要确认以下路径正确：
- `NUPLAN_DATA_ROOT`：nuPlan 数据集路径
- `NUPLAN_MAPS_ROOT`：地图数据路径
- `CONFIG_FILE`：模型 hydra 配置路径
- `CKPT_FILE`：模型 checkpoint 路径

### 3. 启动实验
```bash
# 使用 nohup 后台运行（推荐）
nohup bash scripts/run_w_grid_search.sh > /root/autodl-tmp/w_grid_search.log 2>&1 &

# 查看进度
tail -f /root/autodl-tmp/w_grid_search.log
```

### 4. 预计耗时
| 项目 | 时间 |
|------|------|
| 每个 w 值 | ~3-4 小时 |
| 共 6 个 w 值 | ~18-24 小时 |

脚本支持 **断点续跑**：如果中途中断，重新运行脚本会自动跳过已完成的 w 值。

### 5. 查看结果
全部跑完后，分析报告会自动生成在：
```
/root/autodl-tmp/w_grid_search_output/w_grid_search_report.md
```

也可以手动运行分析：
```bash
conda activate flow_planner
python scripts/analyze_w_by_scenario.py \
    --results_dir /root/autodl-tmp/w_grid_search_output \
    --output_file /root/autodl-tmp/w_grid_search_output/w_grid_search_report.md
```

### 6. 自动关机
脚本跑完后（包括分析）会**自动关机**，不用担心费用。

## 目录结构
```
/root/autodl-tmp/w_grid_search_output/
├── w_0.5/
│   ├── metrics/          # 场景 metric pickle 文件
│   ├── simulation_log/   # 仿真日志
│   └── simulation.log    # 控制台输出
├── w_1.0/
├── w_1.5/
├── w_2.0/
├── w_2.5/
├── w_3.0/
└── w_grid_search_report.md  # 最终分析报告
```

## 常见问题

**Q: 跑了一半中断了怎么办？**
A: 直接重启脚本即可，会自动跳过已完成的 w 值（检查 metrics 文件数 ≥ 1000）。

**Q: 磁盘空间够吗？**
A: 每个 w 的仿真输出约 3-5GB，共 6 个约 18-30GB。确保数据盘剩余 ≥ 40GB。

**Q: 怎么看进度？**
A: `tail -f /root/autodl-tmp/w_grid_search.log` 查看日志，或者直接数 metrics 文件：
```bash
for d in /root/autodl-tmp/w_grid_search_output/w_*/metrics; do
    echo "$(basename $(dirname $d)): $(ls $d 2>/dev/null | wc -l) metrics"
done
```
