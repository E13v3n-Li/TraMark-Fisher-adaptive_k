"""
Adaptive-k 控制器：在不改变总 FL communication rounds 的前提下，
根据 watermark verification rate (VR) 和 main-task accuracy，
把固定的水印容量比例 k 改造成自适应增长（increase-only，第一版不主动释放）。

设计要点（和 server.py 的实际实现对齐）：
- VR / main-task accuracy 都沿用 server.py 里 current_veri / current_acc 的量纲，
  即百分比制(0~100)，不是论文公式里的 0~1 小数，避免单位不一致导致判断永远失效。
- eval_interval 按“post-warmup 的有效训练轮数” * eval_interval_ratio 换算，
  而不是写死绝对轮数，用来对齐不同数据集（例如 fmnist 训练 50 轮、
  其他数据集训练 100 轮）之间的调整机会数，保证实验可比。
- update() 本身只做决策（EMA + 是否需要增大k），不触碰 mask；
  mask 的重新估计、union、以及“actual k”的计算都放在 server.py 里完成，
  完成后再调用 record() 把最终结果（含 actual_k）写入日志和 CSV。
  这样即使某一轮决策后 mask 没有实际增长（new_k == old_k），也不会有
  未完成的日志记录。
"""

import csv
import logging
import os
from dataclasses import dataclass


@dataclass
class AdaptiveKConfig:
    k_init: float = 0.01
    k_min: float = 0.005
    k_max: float = 0.05
    k_step: float = 0.002

    # 百分比制(0~100)，与 server.py 里 current_veri / current_acc 保持一致
    target_vr: float = 95.0
    target_acc: float = 0.0  # 0 表示不启用 accuracy gate

    vr_tolerance: float = 0.5
    acc_tolerance: float = 0.5

    ema_beta: float = 0.8

    # 相对 post-warmup 有效轮数的比例，而不是绝对轮数
    eval_interval_ratio: float = 0.1

    increase_only: bool = True


class AdaptiveKController:
    def __init__(self, config: AdaptiveKConfig, training_rounds: int, warmup_rounds: int, log_dir: str = None):
        self.cfg = config
        self.k = config.k_init
        self.ema_vr = None
        self.ema_acc = None
        self.warmup_rounds = warmup_rounds
        self._k_max_warning_logged = False

        # 按“有效训练轮数”（排除warmup，因为warmup阶段没有mask/没有水印训练）换算出实际的
        # 绝对 eval_interval，保证不同 training_rounds 的数据集之间调整密度一致。
        active_rounds = max(1, training_rounds - warmup_rounds)
        self.eval_interval = max(1, int(round(active_rounds * config.eval_interval_ratio)))

        self.history = []
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, 'adaptive_k_history.csv') if log_dir else None
        if self.csv_path:
            os.makedirs(log_dir, exist_ok=True)
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['round', 'vr', 'ema_vr', 'main_acc', 'ema_acc', 'k_old', 'k_new', 'actual_k'])

        logging.info(
            "[Adaptive-k] controller initialized: k_init=%.4f k_min=%.4f k_max=%.4f k_step=%.4f "
            "target_vr=%.2f target_acc=%.2f eval_interval=%d (active_rounds=%d, ratio=%.3f, warmup_rounds=%d)"
            % (self.k, config.k_min, config.k_max, config.k_step, config.target_vr, config.target_acc,
               self.eval_interval, active_rounds, config.eval_interval_ratio, warmup_rounds)
        )

    def should_update(self, round_id):
        """是否在这一轮做 adaptive-k 决策。warmup 阶段不生效。"""
        if round_id <= self.warmup_rounds:
            return False
        return (round_id - self.warmup_rounds) % self.eval_interval == 0

    def decide(self, round_id, vr, main_acc):
        """
        更新 VR / Acc 的 EMA，并决定新的 k。不改动 mask，也不写日志/CSV
        （因为此时还不知道 mask 扩张后的 actual_k），调用方需要在应用完
        mask 更新后调用 record() 完成记录。

        返回: (new_k, info)，info 里除 actual_k 外的字段均已填好。
        """
        old_k = self.k
        beta = self.cfg.ema_beta
        self.ema_vr = vr if self.ema_vr is None else beta * self.ema_vr + (1 - beta) * vr
        self.ema_acc = main_acc if self.ema_acc is None else beta * self.ema_acc + (1 - beta) * main_acc

        acc_gate_ok = (self.cfg.target_acc <= 0) or (
            self.ema_acc >= self.cfg.target_acc - self.cfg.acc_tolerance
        )

        new_k = old_k
        if self.ema_vr < self.cfg.target_vr - self.cfg.vr_tolerance:
            # 情况 A：VR 不够
            if acc_gate_ok:
                if old_k < self.cfg.k_max:
                    new_k = min(old_k + self.cfg.k_step, self.cfg.k_max)
                else:
                    # 情况 C：已经到 k_max，VR 仍不达标
                    if not self._k_max_warning_logged:
                        logging.warning("[Adaptive-k] WARNING: target VR not reached at k_max")
                        self._k_max_warning_logged = True
            else:
                logging.info(
                    "[Adaptive-k] round=%d: VR未达标但accuracy gate拦截了k的增长 "
                    "(ema_acc=%.4f < target_acc-tol=%.4f)"
                    % (round_id, self.ema_acc, self.cfg.target_acc - self.cfg.acc_tolerance)
                )
        # else: 情况 B，VR已达标，k不变（increase_only，第一版不主动释放watermark参数）

        self.k = new_k

        info = {
            'round': round_id,
            'vr': vr,
            'ema_vr': self.ema_vr,
            'main_acc': main_acc,
            'ema_acc': self.ema_acc,
            'k_old': old_k,
            'k_new': new_k,
        }
        return new_k, info

    def record(self, info, actual_k):
        """在 server.py 完成 mask 扩张（或确认k未变）、算出 actual_k 之后调用，
        写日志和 CSV。"""
        info = dict(info)
        info['actual_k'] = actual_k
        self.history.append(info)

        logging.info(
            "[Adaptive-k] round=%d VR=%.4f EMA_VR=%.4f MainAcc=%.4f EMA_Acc=%.4f k=%.4f -> %.4f (actual_k=%.4f)"
            % (info['round'], info['vr'], info['ema_vr'], info['main_acc'], info['ema_acc'],
               info['k_old'], info['k_new'], actual_k)
        )

        if self.csv_path:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    info['round'],
                    f"{info['vr']:.4f}",
                    f"{info['ema_vr']:.4f}",
                    f"{info['main_acc']:.4f}",
                    f"{info['ema_acc']:.4f}",
                    f"{info['k_old']:.4f}",
                    f"{info['k_new']:.4f}",
                    f"{actual_k:.4f}",
                ])
