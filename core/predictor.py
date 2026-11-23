"""
机制一 · 预测与滚动校准模块（core/predictor.py）
================================================
对应汇报第 6、8 页：
- 轨道几何：确定性外推（直接查表 SGP4 / Walker-Delta 离线预计算轨迹）。
- 路径负载与中断风险：统计外推，不仅给出期望值 (L_hat, O_hat)，还给出随预测步长增长的
  不确定度区间 (sigma_L, sigma_O)。
- 滚动校准：持续比对预测值与实测遥测值，使用 EWMA 更新系统性偏差与残差标准差，
  并对近期发生确认失败/链路故障的卫星施加速衰的短期风险修正（不偷看未来故障）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import math
import numpy as np
from scipy.stats import norm

from env.channel_load import NetworkEnvironment


@dataclass
class ForecastWindow:
    """面向未来 H 步 (t + Δt, ..., t + H·Δt) 的预测输出（带不确定度）。"""

    start_tick: int
    horizon_H: int
    step_ticks: int
    elev_deg: np.ndarray        # (H, N)
    rem_vis_sec: np.ndarray     # (H, N)
    gw_reachable: np.ndarray    # (H, N)
    pred_delay_ms: np.ndarray   # (H, N)
    pred_load: np.ndarray       # (H, N)
    sigma_load: np.ndarray      # (H, N)
    pred_outage: np.ndarray     # (H, N)
    sigma_outage: np.ndarray    # (H, N)


class RollingPredictor:
    """
    结合确定性星历、统计外推与在线残差校准的预测器。
    严格因果：在时刻 tick 只能访问 <= tick 的历史观测，绝不窥视未来真实状态。
    """

    def __init__(
        self,
        env: NetworkEnvironment,
        dt_plan: float,
        ewma_alpha: float = 0.25,
        enable_calibration: bool = True,
    ) -> None:
        self.env = env
        self.orbit = env.orbit
        self.dt_sim = env.orbit.dt_sim
        self.dt_plan = dt_plan
        self.step_ticks = max(1, int(round(dt_plan / self.dt_sim)))
        self.ewma_alpha = ewma_alpha
        self.enable_calibration = enable_calibration

        N = len(self.orbit.sat_ids)
        # 预测器初始先验均值存在系统性低估偏差（测试滚动校准机制是否能纠偏）
        self.prior_mean = np.clip(env.base_mean_load - env.pred_bias_drift, 0.08, 0.85).astype(np.float32)
        self.phi = float(env.ar1_phi)
        self.base_sigma = float(env.ar1_sigma_step)

        # 在线滚动校准状态
        self.bias_est = np.zeros(N, dtype=np.float32)          # 逐星负载偏差估计 b_s
        self.global_bias = 0.0                                 # 全局偏差估计（缓解新星冷启动）
        self.var_est = np.full(N, self.base_sigma ** 2, dtype=np.float32)
        self.obs_count = np.zeros(N, dtype=np.int32)
        self.last_obs_load = self.prior_mean.copy()
        self.last_pred_one_step = self.prior_mean.copy()

        # 短期异常风险记忆（如某星刚发生雨衰/硬故障/确认失败，按时间衰减）
        self.anomaly_risk = np.zeros(N, dtype=np.float32)
        self.last_update_tick = 0

    def observe_and_calibrate(
        self,
        tick: int,
        serving_sat: Optional[int] = None,
        failed_sat: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        使用当前时刻的遥测观测更新预测器偏差与不确定度模型。
        返回当前服务星的实测与预测残差，供机制四监控模块使用。
        """
        dt_elapsed = max(0.0, (tick - self.last_update_tick) * self.dt_sim)
        self.last_update_tick = tick
        # 异常风险记忆指数衰减（半衰期约 18 s）
        if dt_elapsed > 0:
            self.anomaly_risk *= math.exp(-dt_elapsed / 25.0)

        # 地面信关站可周期性收集当前所有可见卫星的轻量负载遥测
        vis_mask = self.orbit.elevation_deg[tick] >= (self.env.min_elev_deg - 3.0)
        obs_load = self.env.true_load[tick]

        # 计算与上一次单步预测的残差
        residual = obs_load - self.last_pred_one_step

        if self.enable_calibration:
            alpha = self.ewma_alpha
            idx = np.where(vis_mask)[0]
            if len(idx) > 0:
                self.bias_est[idx] = (1.0 - alpha) * self.bias_est[idx] + alpha * residual[idx]
                err_sq = (residual[idx] - self.bias_est[idx]) ** 2
                self.var_est[idx] = np.clip(
                    (1.0 - alpha) * self.var_est[idx] + alpha * err_sq,
                    (0.4 * self.base_sigma) ** 2,
                    0.28 ** 2,
                )
                self.obs_count[idx] += 1
                self.global_bias = (1.0 - 0.15) * self.global_bias + 0.15 * float(np.mean(residual[idx]))

        self.last_obs_load = np.where(vis_mask, obs_load, self.last_obs_load)

        # 更新下一次一步预测基准（供下个监控步比对残差）
        calib_mu = self._calibrated_mean()
        self.last_pred_one_step = np.clip(
            calib_mu + self.phi * (self.last_obs_load - calib_mu),
            0.05,
            0.98,
        )

        # 若观测到某星当前链路中断/弱信噪比，或切换确认失败，立即提高该星短期风险记忆
        if serving_sat is not None and 0 <= serving_sat < len(self.anomaly_risk):
            if self.env.link_outage_raw[tick, serving_sat]:
                self.anomaly_risk[serving_sat] = min(0.95, max(self.anomaly_risk[serving_sat], 0.78))
            elif self.env.effective_snr_db[tick, serving_sat] < self.env.snr_threshold_db + 1.5:
                self.anomaly_risk[serving_sat] = min(0.90, max(self.anomaly_risk[serving_sat], 0.45))

        if failed_sat is not None and 0 <= failed_sat < len(self.anomaly_risk):
            self.anomaly_risk[failed_sat] = 0.92

        # 输出服务星当前偏差与对应不确定度，供机制四软触发判定
        info = {"serving_residual": 0.0, "serving_sigma": self.base_sigma, "serving_load": 0.0}
        if serving_sat is not None and 0 <= serving_sat < len(obs_load):
            info["serving_residual"] = float(obs_load[serving_sat] - self.last_pred_one_step[serving_sat])
            info["serving_sigma"] = float(math.sqrt(self.var_est[serving_sat]))
            info["serving_load"] = float(obs_load[serving_sat])
        return info

    def _calibrated_mean(self) -> np.ndarray:
        """融合单星校准偏差与全局冷启动偏差，得到校准后的长期均值估计。"""
        if not self.enable_calibration:
            return self.prior_mean
        # 观测次数少的卫星更多借用全局偏差估计（应对第 16 页冷启动边界）
        w_local = np.minimum(self.obs_count / 6.0, 1.0).astype(np.float32)
        eff_bias = w_local * self.bias_est + (1.0 - w_local) * self.global_bias
        return np.clip(self.prior_mean + eff_bias, 0.08, 0.92)

    def forecast(self, tick: int, horizon_H: int) -> ForecastWindow:
        """
        生成未来 H 个规划步 (t + Δt ... t + H·Δt) 的预测矩阵与不确定度矩阵。
        """
        T, N = self.orbit.elevation_deg.shape
        elev = np.zeros((horizon_H, N), dtype=np.float32)
        rem_vis = np.zeros((horizon_H, N), dtype=np.float32)
        gw_reach = np.zeros((horizon_H, N), dtype=bool)
        pred_delay = np.zeros((horizon_H, N), dtype=np.float32)
        pred_load = np.zeros((horizon_H, N), dtype=np.float32)
        sigma_load = np.zeros((horizon_H, N), dtype=np.float32)
        pred_outage = np.zeros((horizon_H, N), dtype=np.float32)
        sigma_outage = np.zeros((horizon_H, N), dtype=np.float32)

        calib_mu = self._calibrated_mean()
        # 当前时刻观测到的负载（不可见星退回校准均值）
        vis_now = self.orbit.elevation_deg[tick] >= (self.env.min_elev_deg - 3.0)
        cur_load = np.where(vis_now, self.env.true_load[tick], calib_mu)

        # 冷启动不确定度膨胀因子：观测越少，不确定度 σ 越大
        cold_factor = np.where(self.obs_count < 4, 1.35, 1.0).astype(np.float32)
        sigma_step = np.sqrt(self.var_est) * cold_factor

        for h in range(1, horizon_H + 1):
            ft = min(T - 1, tick + h * self.step_ticks)
            el_h = self.orbit.elevation_deg[ft]
            rv_h = self.orbit.rem_vis_sec[ft]
            gw_h = self.orbit.gw_reachable[ft]
            prop_h = self.orbit.prop_delay_ms[ft]

            # 1. AR(1) 统计外推：期望向均值回归，不确定度随步长 h 单调增大
            phi_h = self.phi ** h
            mu_h = np.clip(calib_mu + phi_h * (cur_load - calib_mu), 0.05, 0.98)
            var_scale = math.sqrt((1.0 - self.phi ** (2 * h)) / max(1.0 - self.phi ** 2, 1e-6))
            sig_l_h = np.clip(sigma_step * var_scale, 0.03, 0.35)

            # 2. 预测端到端时延（含 M/M/1 排队时延）
            q_ms = 9.0 * mu_h / np.maximum(1.0 - np.minimum(mu_h, 0.95), 0.05)
            d_h = np.where(gw_h, prop_h + q_ms, 999.0)

            # 3. 预测中断概率 Ô_t：由三部分风险合成
            #    (a) 低仰角边缘雨衰/遮挡风险（仰角越接近 θ_min，受雨衰与快衰落击穿概率越高）
            el_margin = el_h - self.env.min_elev_deg
            p_edge = 1.0 / (1.0 + np.exp(0.65 * (el_margin - 2.0)))
            p_edge = np.where(el_h < self.env.min_elev_deg, 1.0, p_edge * 0.25)

            #    (b) 过载尾风险 P(L_{t+h} >= 0.89)
            z_cong = (0.89 - mu_h) / np.maximum(sig_l_h, 0.02)
            p_cong = 1.0 - norm.cdf(z_cong)

            #    (c) 短期异常记忆衰减 + 极小本底随机故障率
            p_anom = self.anomaly_risk * math.exp(-(h * self.dt_plan) / 22.0)
            p_base = 0.001

            # 独立风险合成：1 - Π(1 - p_i)
            p_out = 1.0 - (1.0 - np.clip(p_edge, 0.0, 1.0)) * (1.0 - np.clip(p_cong, 0.0, 1.0)) * (
                1.0 - np.clip(p_anom, 0.0, 0.95)
            ) * (1.0 - p_base)
            p_out = np.where((el_h < self.env.min_elev_deg) | (~gw_h), 0.995, np.clip(p_out, 0.0005, 0.995))

            # 中断风险自身的不确定度 σ_O（随步长增长，且在边缘仰角/高负载波动区更大）
            sig_o_h = np.clip(0.008 * math.sqrt(h) + 0.30 * p_out * (1.0 - p_out) + 0.20 * sig_l_h, 0.005, 0.30)

            elev[h - 1] = el_h
            rem_vis[h - 1] = rv_h
            gw_reach[h - 1] = gw_h
            pred_delay[h - 1] = d_h
            pred_load[h - 1] = mu_h
            sigma_load[h - 1] = sig_l_h
            pred_outage[h - 1] = p_out
            sigma_outage[h - 1] = sig_o_h

        return ForecastWindow(
            start_tick=tick,
            horizon_H=horizon_H,
            step_ticks=self.step_ticks,
            elev_deg=elev,
            rem_vis_sec=rem_vis,
            gw_reachable=gw_reach,
            pred_delay_ms=pred_delay,
            pred_load=pred_load,
            sigma_load=sigma_load,
            pred_outage=pred_outage,
            sigma_outage=sigma_outage,
        )
