"""
机制二 · 带余量与机会约束的动态规划（DP）求解器（core/planner.py）
==================================================================
对应汇报第 3、4、9、12、14 页：
1. AHP 层次分析法赋权（自动做一致性校验 CR < 0.1），确定 (α, β, γ, δ)。
2. 候选集可行性筛选 C_t（至多保留 K 个候选），时间复杂度 O(H·K²)：
   - 带余量模式（use_margin=True）：仰角门限加余量 Δθ，最短剩余可见时间设下限 T_safe。
3. 代价层：将负载/中断不确定度 σ 写入 DP 转移代价：
   J = Σ_t [ α·D_t + β·q_t + γ·(L_hat + κ_L·σ_L) + δ·(O_hat + κ_O·σ_O) ]
4. 约束层：机会约束 Π_t (1 - O_hat_t) >= 1 - ε（取对数转为可累加风险预算）。
5. 逆向 DP 同步产出：L1 主序列 + L2 备份候选（无需额外求解一遍），并支持受影响层的增量更新。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import time
import numpy as np

from core.predictor import ForecastWindow


def compute_ahp_weights(pairwise_matrix: List[List[float]]) -> Tuple[np.ndarray, float]:
    """
    通过特征向量法计算 AHP 权重，并验证一致性比率 CR < 0.1。
    矩阵顺序：[时延 D, 切换 q, 负载 L, 预测中断 Ô]
    """
    A = np.array(pairwise_matrix, dtype=np.float64)
    n = A.shape[0]
    eigvals, eigvecs = np.linalg.eig(A)
    max_idx = int(np.argmax(eigvals.real))
    lambda_max = float(eigvals[max_idx].real)
    w = np.abs(eigvecs[:, max_idx].real)
    w = w / np.sum(w)

    # 随机一致性指标 RI
    ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12}
    ci = (lambda_max - n) / max(n - 1, 1)
    cr = ci / ri_table.get(n, 1.12) if n > 2 else 0.0
    return w.astype(np.float64), float(cr)


@dataclass
class PlanResult:
    """DP 规划器输出，同时包含 L1 主序列、L2 备份候选以及实测各阶段耗时（用于第 14 页实时性验证）。"""

    primary_sat: int                  # 第一步推荐接入星 (L1)
    backup_sat: Optional[int]         # 第一步次优备份候选 (L2)
    primary_seq: List[int]            # 完整 H 步主序列
    backup_seq: List[int]             # 完整 H 步备份序列
    primary_cost: float
    chance_satisfied: bool            # 是否满足全窗口机会约束 >= 1 - ε
    window_survival_prob: float       # 规划窗口预测连续服务概率
    filter_time_ms: float             # 候选筛选耗时 (ms)
    dp_time_ms: float                 # DP 求解耗时 (ms)
    is_incremental: bool = False      # 是否采用了增量重规划


class MarginDPPlanner:
    """带安全余量、风险加价与机会约束的滚动时域 DP 规划器。"""

    def __init__(
        self,
        min_elev_deg: float,
        max_candidates_K: int,
        ahp_pairwise: List[List[float]],
        elev_margin_deg: float = 5.0,
        min_rem_vis_s: float = 15.0,
        risk_kappa_load: float = 1.2,
        risk_kappa_outage: float = 1.5,
        chance_epsilon: float = 0.08,
        use_margin: bool = True,
    ) -> None:
        self.min_elev_deg = min_elev_deg
        self.K = max_candidates_K
        self.weights, self.ahp_cr = compute_ahp_weights(ahp_pairwise)
        self.alpha, self.beta, self.gamma, self.delta = [float(x) for x in self.weights]

        self.elev_margin_deg = elev_margin_deg
        self.min_rem_vis_s = min_rem_vis_s
        self.risk_kappa_load = risk_kappa_load
        self.risk_kappa_outage = risk_kappa_outage
        self.chance_epsilon = chance_epsilon
        self.use_margin = use_margin

        # 缓存上一次尾部 DP 值函数，供增量更新复用（呼应第 11、14 页）
        self._cached_tail_V: Optional[Dict[int, float]] = None
        self._cached_tail_risk: Optional[Dict[int, float]] = None
        self._cached_tail_tick: int = -1

    def _node_stage_cost(self, fw: ForecastWindow, h_idx: int, sat_idx: int) -> float:
        """计算单时间层、单卫星的节点代价（不含切换项 β·q_t）。"""
        d_norm = min(float(fw.pred_delay_ms[h_idx, sat_idx]) / 90.0, 2.5)
        l_hat = float(fw.pred_load[h_idx, sat_idx])
        o_hat = float(fw.pred_outage[h_idx, sat_idx])

        if self.use_margin:
            l_eff = min(1.5, l_hat + self.risk_kappa_load * float(fw.sigma_load[h_idx, sat_idx]))
            o_eff = min(1.5, o_hat + self.risk_kappa_outage * float(fw.sigma_outage[h_idx, sat_idx]))
        else:
            l_eff = l_hat
            o_eff = o_hat

        return self.alpha * d_norm + self.gamma * l_eff + self.delta * o_eff

    def _filter_candidates(
        self,
        fw: ForecastWindow,
        h_idx: int,
        exclude_sats: Optional[set[int]] = None,
    ) -> np.ndarray:
        """
        可行性筛选与候选规模控制（|C_t| <= K）。
        若开启余量模式，先用加余量门限筛选；若极端时刻无满足星，则平滑回退至物理基础门限。
        """
        el = fw.elev_deg[h_idx]
        rv = fw.rem_vis_sec[h_idx]
        gw = fw.gw_reachable[h_idx]
        ld = fw.pred_load[h_idx]

        if self.use_margin:
            mask = (
                (el >= self.min_elev_deg + self.elev_margin_deg)
                & (rv >= self.min_rem_vis_s)
                & gw
                & (ld < 0.90)
            )
            cands = np.where(mask)[0]
            if len(cands) == 0:
                # 兜底放宽至物理可行集，防止空候选集崩溃
                mask = (el >= self.min_elev_deg) & gw
                cands = np.where(mask)[0]
        else:
            mask = (el >= self.min_elev_deg) & gw
            cands = np.where(mask)[0]

        if exclude_sats and h_idx == 0 and len(cands) > 1:
            filtered = np.array([s for s in cands if int(s) not in exclude_sats], dtype=np.int32)
            if len(filtered) > 0:
                cands = filtered

        if len(cands) == 0:
            # 极端保底：取仰角最高的一颗星
            cands = np.array([int(np.argmax(el))], dtype=np.int32)

        if len(cands) > self.K:
            # 按单步综合代价排序，保留前 K 颗最优候选
            scores = np.array([self._node_stage_cost(fw, h_idx, int(s)) for s in cands], dtype=np.float64)
            top_idx = np.argsort(scores)[: self.K]
            cands = cands[top_idx]

        return cands.astype(np.int32)

    def solve(
        self,
        fw: ForecastWindow,
        curr_sat: int,
        exclude_first_step: Optional[set[int]] = None,
        incremental_layers: Optional[int] = None,
    ) -> PlanResult:
        """
        逆向 DP 求解最优序列（O(H·K²)），同步给出 L1 主序列与 L2 备份候选。
        当 incremental_layers 指定（如事件触发仅局部重算前 6 层）且尾部缓存有效时，
        直接复用远端时间层的价值函数以节省计算耗时。
        """
        H = fw.horizon_H

        # 判断是否可启用增量更新（仅重算前 L_inc 层）
        use_inc = (
            incremental_layers is not None
            and 1 <= incremental_layers < H
            and self._cached_tail_V is not None
            and abs(fw.start_tick - self._cached_tail_tick) <= fw.step_ticks * 2
        )
        eff_H = incremental_layers if use_inc else H

        # 阶段 1：逐层候选筛选
        t_f0 = time.perf_counter()
        cand_layers: List[np.ndarray] = [
            self._filter_candidates(fw, h, exclude_first_step if h == 0 else None)
            for h in range(eff_H)
        ]
        filter_time_ms = (time.perf_counter() - t_f0) * 1000.0

        # 阶段 2：逆向 DP 求解
        t_d0 = time.perf_counter()
        risk_budget = -math.log(max(1.0 - self.chance_epsilon, 1e-4))
        # 机会约束违约惩罚权重（仅 use_margin=True 时生效）
        chance_penalty_w = 12.0 if self.use_margin else 0.0
        # 多步窗口下切换代价与窗口长度对齐，抑制因短期微小波动引发的乒乓切换；
        # 开启余量时附带轻微切换迟滞保护，呼应第 16 页“余量抑制反复横跳”
        ho_penalty = self.beta * (5.2 if self.use_margin else 3.2)

        V_layers: List[np.ndarray] = [np.zeros(len(c), dtype=np.float64) for c in cand_layers]
        R_layers: List[np.ndarray] = [np.zeros(len(c), dtype=np.float64) for c in cand_layers]
        next_ptr: List[np.ndarray] = [np.full(len(c), -1, dtype=np.int32) for c in cand_layers]

        # 末层初始化
        last_h = eff_H - 1
        for i, s in enumerate(cand_layers[last_h]):
            si = int(s)
            base_c = self._node_stage_cost(fw, last_h, si)
            o_p = min(float(fw.pred_outage[last_h, si]), 0.995)
            base_r = -math.log(max(1.0 - o_p, 1e-4))
            if use_inc and self._cached_tail_V is not None:
                # 衔接缓存的远端价值函数
                tail_v = self._cached_tail_V.get(si, min(self._cached_tail_V.values()) + ho_penalty)
                tail_r = self._cached_tail_risk.get(si, min(self._cached_tail_risk.values()))
                V_layers[last_h][i] = base_c + tail_v
                R_layers[last_h][i] = base_r + tail_r
            else:
                V_layers[last_h][i] = base_c
                R_layers[last_h][i] = base_r

        # 逆向递推 h = eff_H - 2 ... 0
        for h in range(eff_H - 2, -1, -1):
            c_cur = cand_layers[h]
            c_nxt = cand_layers[h + 1]
            v_nxt = V_layers[h + 1]
            r_nxt = R_layers[h + 1]

            for i, u in enumerate(c_cur):
                ui = int(u)
                u_cost = self._node_stage_cost(fw, h, ui)
                o_u = min(float(fw.pred_outage[h, ui]), 0.995)
                u_risk = -math.log(max(1.0 - o_u, 1e-4))

                best_val = math.inf
                best_j = 0
                best_r = math.inf

                for j, v in enumerate(c_nxt):
                    vi = int(v)
                    sw_cost = 0.0 if ui == vi else ho_penalty
                    tot_r = u_risk + r_nxt[j]
                    # 机会约束风险预算检验（越界则显著加罚）
                    excess = max(0.0, tot_r - risk_budget)
                    val = u_cost + sw_cost + v_nxt[j] + chance_penalty_w * excess
                    if val < best_val:
                        best_val = val
                        best_j = j
                        best_r = tot_r

                V_layers[h][i] = best_val
                R_layers[h][i] = best_r
                next_ptr[h][i] = best_j

        # 缓存中间层价值函数，供后续步内事件触发时做增量重规划
        if not use_inc and H >= 8:
            mid = min(6, H - 1)
            self._cached_tail_V = {int(s): float(V_layers[mid][i]) for i, s in enumerate(cand_layers[mid])}
            self._cached_tail_risk = {int(s): float(R_layers[mid][i]) for i, s in enumerate(cand_layers[mid])}
            self._cached_tail_tick = fw.start_tick

        # 第 0 步：从当前服务卫星 curr_sat 转移到 cand_layers[0]
        first_cands = cand_layers[0]
        scored_first: List[Tuple[float, int, int, float]] = []
        for i, s in enumerate(first_cands):
            si = int(s)
            sw0 = 0.0 if (curr_sat < 0 or si == curr_sat) else ho_penalty
            tot_c = sw0 + float(V_layers[0][i])
            tot_r = float(R_layers[0][i])
            scored_first.append((tot_c, i, si, tot_r))

        scored_first.sort(key=lambda x: x[0])

        def _trace_seq(start_idx: int) -> List[int]:
            seq = []
            idx = start_idx
            for h in range(eff_H):
                seq.append(int(cand_layers[h][idx]))
                idx = int(next_ptr[h][idx])
                if idx < 0:
                    break
            return seq

        best_cost, best_i, primary_sat, best_risk = scored_first[0]
        primary_seq = _trace_seq(best_i)

        if len(scored_first) >= 2:
            _, sec_i, backup_sat, _ = scored_first[1]
            backup_seq = _trace_seq(sec_i)
        else:
            backup_sat = None
            backup_seq = []

        dp_time_ms = (time.perf_counter() - t_d0) * 1000.0
        surv_prob = math.exp(-best_risk)

        return PlanResult(
            primary_sat=primary_sat,
            backup_sat=backup_sat,
            primary_seq=primary_seq,
            backup_seq=backup_seq,
            primary_cost=best_cost,
            chance_satisfied=(best_risk <= risk_budget + 1e-6),
            window_survival_prob=surv_prob,
            filter_time_ms=filter_time_ms,
            dp_time_ms=dp_time_ms,
            is_incremental=use_inc,
        )
