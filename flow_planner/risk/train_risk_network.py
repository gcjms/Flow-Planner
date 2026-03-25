"""
训练 Risk Network
==================
使用 Grid Search 产生的 (风险特征, 最优w) 数据训练 Risk MLP。

用法:
    python -m flow_planner.risk.train_risk_network \
        --dataset risk_dataset.npz \
        --output risk_network.pth \
        --epochs 200
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from flow_planner.risk.risk_network import RiskNetwork
from flow_planner.risk.risk_features import NUM_RISK_FEATURES, RISK_FEATURE_NAMES, normalize_features


def train_risk_network(
    features: np.ndarray,
    optimal_w: np.ndarray,
    output_path: str,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    w_min: float = 0.5,
    w_max: float = 4.0,
    val_split: float = 0.2,
    device: str = 'cpu',
):
    """
    训练 Risk Network。
    
    Args:
        features: (N, 12) 风险特征
        optimal_w: (N,) Grid Search 找到的最优 w
        output_path: 保存路径
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        hidden_dim: 隐藏层维度
        w_min / w_max: CFG 权重范围
        val_split: 验证集比例
        device: 训练设备
    """
    print(f"Training Risk Network")
    print(f"  Samples: {len(features)}")
    print(f"  Features: {NUM_RISK_FEATURES}")
    print(f"  w range: [{w_min}, {w_max}]")
    print(f"  Device: {device}")
    
    # ---- 归一化特征 ----
    features_norm, norm_stats = normalize_features(features)
    
    # ---- 将 w 转换为 risk_score (归一化到 [0,1]) ----
    risk_targets = (optimal_w - w_min) / (w_max - w_min)
    risk_targets = np.clip(risk_targets, 0.0, 1.0)
    
    # ---- 构建数据集 ----
    X = torch.from_numpy(features_norm).float()
    Y = torch.from_numpy(risk_targets).float().unsqueeze(-1)
    
    dataset = TensorDataset(X, Y)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    print(f"  Train: {train_size}, Val: {val_size}")
    
    # ---- 构建模型 ----
    model = RiskNetwork(
        input_dim=NUM_RISK_FEATURES,
        hidden_dim=hidden_dim,
        w_min=w_min,
        w_max=w_max,
    ).to(device)
    
    # 设置归一化参数（使用已归一化的数据，所以内部归一化设为恒等）
    model.feature_mean = torch.zeros(NUM_RISK_FEATURES).to(device)
    model.feature_std = torch.ones(NUM_RISK_FEATURES).to(device)
    
    print(f"  Model parameters: {model.num_parameters}")
    
    # ---- 训练 ----
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            output = model(X_batch)
            risk_pred = output['risk_score']
            
            loss = criterion(risk_pred, Y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(X_batch)
        
        epoch_loss /= train_size
        train_losses.append(epoch_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output['risk_score'], Y_batch)
                val_loss += loss.item() * len(X_batch)
        
        val_loss /= val_size
        val_losses.append(val_loss)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                'model_state_dict': model.state_dict(),
                'input_dim': NUM_RISK_FEATURES,
                'hidden_dim': hidden_dim,
                'w_min': w_min,
                'w_max': w_max,
                'feature_mean': norm_stats['mean'],
                'feature_std': norm_stats['std'],
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'epoch': epoch,
            }
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}: "
                  f"train_loss={epoch_loss:.6f}, val_loss={val_loss:.6f}, "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")
    
    # ---- 保存最佳模型 ----
    best_state['train_losses'] = np.array(train_losses)
    best_state['val_losses'] = np.array(val_losses)
    torch.save(best_state, output_path)
    
    print(f"\nBest model saved to {output_path}")
    print(f"  Best epoch: {best_state['epoch']+1}")
    print(f"  Best val loss: {best_state['val_loss']:.6f}")
    
    # ---- 分析预测结果 ----
    model.load_state_dict(best_state['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        X_all = torch.from_numpy(features_norm).float().to(device)
        output = model(X_all)
        pred_w = output['w'].cpu().numpy().flatten()
        pred_risk = output['risk_score'].cpu().numpy().flatten()
    
    print(f"\nPrediction analysis:")
    print(f"  Predicted w: mean={pred_w.mean():.2f}, std={pred_w.std():.2f}, "
          f"range=[{pred_w.min():.2f}, {pred_w.max():.2f}]")
    print(f"  Target w:    mean={optimal_w.mean():.2f}, std={optimal_w.std():.2f}, "
          f"range=[{optimal_w.min():.2f}, {optimal_w.max():.2f}]")
    
    # w 预测误差
    w_error = np.abs(pred_w - optimal_w)
    print(f"  MAE(w): {w_error.mean():.3f}")
    print(f"  Within ±0.5 of target: {(w_error <= 0.5).mean()*100:.1f}%")
    print(f"  Within ±1.0 of target: {(w_error <= 1.0).mean()*100:.1f}%")
    
    return model, best_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Risk Network')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Path to risk_dataset.npz (from grid_search_w.py)')
    parser.add_argument('--output', type=str, default='risk_network.pth',
                        help='Output checkpoint path')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--w_min', type=float, default=0.5)
    parser.add_argument('--w_max', type=float, default=4.0)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    # 加载 Grid Search 数据
    data = np.load(args.dataset)
    features = data['features']
    optimal_w = data['optimal_w']
    
    print(f"Loaded {len(features)} samples from {args.dataset}")
    
    train_risk_network(
        features=features,
        optimal_w=optimal_w,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        w_min=args.w_min,
        w_max=args.w_max,
        device=args.device,
    )
