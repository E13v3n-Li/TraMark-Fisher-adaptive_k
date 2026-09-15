import argparse

def get_parse_args():
    parser = argparse.ArgumentParser(description='pass in a parameter')
    
    # System settings
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_path', type=str, default='/home/jiahaox/TraMark/data', help='the path to the dataset')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--client_sampling_ratio', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)

    # Local training settings
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--local_bs', type=int, default=64)
    parser.add_argument('--local_lr', type=float, default=0.01)

    # Non-iid setting
    parser.add_argument('--non_iid', action='store_true', default=False, help='whether to simulate non-iid data distribution, if set to True, please set --gamma')
    parser.add_argument('--gamma', type=float, default=0.5, help='the parameter for non-iid data distribution, the smaller the more non-iid')
    
    # TraMark settings
    parser.add_argument('--watermarking_method', type=str, default='tramark', help='the watermarking method to be used')
    parser.add_argument('--alpha', type=float, default=0.1, help='the fraction of warmup training rounds')
    parser.add_argument('--k', type=float, default=0.01, help='the partition ratio for the main task region and the watermarking task region')

    # 参数重要性度量方式（对应"阶段A"消融实验）
    parser.add_argument('--importance_method', type=str, default='topk', choices=['topk', 'grad', 'fisher', 'taylor'],
                         help='watermark region怎么划分：topk=原版TraMark(参数绝对值), '
                              'grad=梯度绝对值均值, fisher=Fisher信息(梯度平方,推荐), taylor=一阶泰勒展开重要性')
    parser.add_argument('--importance_num_batches', type=int, default=5,
                         help='用多少个batch的服务器侧公开测试数据来估计梯度类重要性，仅在importance_method!=topk时生效')

    # Adaptive-k settings（在固定 k 的基础上，根据 VR / main-task accuracy 自适应调整 watermark 容量）
    parser.add_argument('--adaptive_k', action='store_true', default=False,
                         help='是否启用 adaptive watermark ratio；默认False，行为与原TraMark完全一致')
    parser.add_argument('--k_min', type=float, default=0.005, help='k 的下界')
    parser.add_argument('--k_max', type=float, default=0.05, help='k 的上界')
    parser.add_argument('--k_step', type=float, default=0.002, help='每次触发调整时 k 的增量')
    parser.add_argument('--target_vr', type=float, default=95.0,
                         help='VR 目标值，百分比制(0~100)，与 server.py 里 current_veri 的量纲保持一致')
    parser.add_argument('--target_acc', type=float, default=0.0,
                         help='主任务accuracy下限，百分比制(0~100)；0表示不启用accuracy gate')
    parser.add_argument('--vr_tolerance', type=float, default=0.5, help='VR 容差，百分比制')
    parser.add_argument('--acc_tolerance', type=float, default=0.5, help='accuracy 容差，百分比制')
    parser.add_argument('--k_ema_beta', type=float, default=0.8, help='VR/Acc 做EMA平滑时的beta')
    parser.add_argument('--k_eval_interval_ratio', type=float, default=0.1,
                         help='每隔 (training_rounds-warmup_rounds)*ratio 轮做一次adaptive-k决策；'
                              '按post-warmup的有效轮数换算，而不是绝对轮数，用来对齐不同数据集'
                              '（如fmnist 50轮 vs 其他100轮）之间的调整密度')

    args = parser.parse_args()
    
    return args