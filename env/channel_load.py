"""
动态负载、信道衰减、端到端时延与可控故障注入模块（统计性 + 随机性分量）
=======================================================================
对应汇报第 2、5、6、15 页：
- 统计性分量：时间相关的外生卫星负载 L_t(s)（AR(1) + 周期波动 + 星间异质性）
- 随机性分量：可控故障注入器（雨衰/遮挡、突发负载尖峰、链路/切换故障、系统性预测偏差）
- 真实端到端时延 D_t 与真实逐时刻中断事件 O_t ∈ {0, 1}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from env.orbit import OrbitTrace


@dataclass
class InjectedFault:
    """记录单个注入故障事件的元信息，用于可视化时间轴回放与归因分析。"""

    fault_type: str        # "rain_fade" | "load_surge" | "link_fault"
    sat_idx: int
    sat_id: str
    start_s: float
    end_s: float
    magnitude: float
    description: str


@dataclass
class NetworkEnvironment:
    """
    存储某一场景下的全部真实物理状态序列（所有基线与消融策略共享同一份真实环境，
    确保绝对公平对比）。
    """

    orbit: OrbitTrace
    scenario_name: str
    seed: int
    min_elev_deg: float
    true_load: np.ndarray           # (T, N) 真实路径负载 L_t ∈ [0.05, 0.99]
    base_mean_load: np.ndarray      # (N,)   各卫星基础平均负载 μ_s
    ar1_phi: float                  # AR(1) 步长自相关系数 (对应 dt_plan 尺度)
    ar1_sigma_step: float           # 过程噪声标准差
    rain_fade_db: np.ndarray        # (T, N) 额外雨衰/遮挡衰减 (dB)
    hard_fault_mask: np.ndarray     # (T, N) 突发硬故障/切换拒绝掩码 (bool)
    effective_snr_db: np.ndarray    # (T, N) 实际有效信噪比 (dB)
    snr_threshold_db: float         # 解调门限 (dB)
    e2e_delay_ms: np.ndarray        # (T, N) 实际端到端时延 D_t (ms)
    link_outage_raw: np.ndarray     # (T, N) 仅由物理链路/过载/故障导致的中断 (bool)
    pred_bias_drift: float          # 注入给预测器先验的系统性低估偏差
    faults: List[InjectedFault] = field(default_factory=list)

    def is_sat_healthy_for_confirm(self, tick: int, sat_idx: int) -> bool:
        """确认式切换（机制三）握手探测：目标卫星当下是否满足稳定接管条件。"""
        if sat_idx < 0 or sat_idx >= len(self.orbit.sat_ids):
            return False
        if self.orbit.elevation_deg[tick, sat_idx] < self.min_elev_deg:
            return False
        if not self.orbit.gw_reachable[tick, sat_idx]:
            return False
        if self.hard_fault_mask[tick, sat_idx]:
            return False
        if self.effective_snr_db[tick, sat_idx] < self.snr_threshold_db:
            return False
        if self.true_load[tick, sat_idx] >= 0.92:
            return False
        return True

    def is_link_degraded(self, tick: int, sat_idx: int) -> bool:
        """判断当前链路是否处于严重拥塞/弱信号降质状态（触发 L4 业务降级）。"""
        if self.link_outage_raw[tick, sat_idx]:
            return True
        if self.true_load[tick, sat_idx] >= 0.84:
            return True
        if self.effective_snr_db[tick, sat_idx] < self.snr_threshold_db + 2.0:
            return True
        return False


def _clear_sky_snr_db(elev_deg: np.ndarray, slant_km: np.ndarray) -> np.ndarray:
    """
    简化的低轨链路预算：高仰角斜距短、大气损耗小，晴空 SNR 更高。
    25° 约 6.8 dB，30° 约 8.6 dB，45° 约 12.8 dB，70° 约 16.5 dB。
    """
    el_rad = np.radians(np.clip(elev_deg, 1.0, 90.0))
    # 自由空间路径损耗相对参考斜距 (700 km) 的变化 + 低仰角大气吸收
    fspl_delta = 20.0 * np.log10(np.maximum(slant_km, 500.0) / 700.0)
    atm_loss = 1.6 / np.sin(el_rad) - 1.6
    snr = 15.5 - fspl_delta - atm_loss
    return np.where(elev_deg > 0.0, snr, -20.0)


def build_environment(
    orbit: OrbitTrace,
    scenario_cfg: Dict,
    sim_cfg: Dict,
) -> NetworkEnvironment:
    """
    构建包含真实动态负载、信道快衰落、端到端时延及可控故障注入的仿真环境。
    """
    seed = int(scenario_cfg["seed"])
    rng = np.random.default_rng(seed)

    T, N = orbit.elevation_deg.shape
    dt_sim = orbit.dt_sim
    dt_plan = float(sim_cfg["dt_plan"])
    duration_s = float(sim_cfg["duration_s"])
    min_elev_deg = float(sim_cfg["min_elev_deg"])

    # 1. 生成各卫星异质基础负载 μ_s ∈ [0.22, 0.76]
    #    高负载星与低负载星混合，自然形成“高仰角但拥塞 vs 中仰角且通畅”的选星冲突
    base_mean_load = rng.uniform(0.22, 0.76, size=N).astype(np.float32)

    # 2. 时间相关的 AR(1) 负载过程（在微步长上平滑演化）
    phi_plan = 0.82
    phi_sim = phi_plan ** (dt_sim / dt_plan)
    sigma_plan = 0.075
    sigma_sim = sigma_plan * np.sqrt((1.0 - phi_sim ** 2) / (1.0 - phi_plan ** 2))

    true_load = np.zeros((T, N), dtype=np.float32)
    true_load[0] = np.clip(base_mean_load + rng.normal(0.0, 0.06, size=N), 0.08, 0.88)

    # 缓慢潮汐波动（模拟区域业务量变化）
    phase = rng.uniform(0.0, 2.0 * np.pi, size=N)
    for t in range(1, T):
        tide = 0.05 * np.sin(2.0 * np.pi * (orbit.times_s[t] / 600.0) + phase)
        target_mu = np.clip(base_mean_load + tide, 0.10, 0.85)
        eps = rng.normal(0.0, sigma_sim, size=N)
        true_load[t] = np.clip(target_mu + phi_sim * (true_load[t - 1] - target_mu) + eps, 0.05, 0.98)

    # 3. 故障注入器：在活跃可见卫星上精准注入雨衰/遮挡、负载突变与突发硬故障
    rain_fade_db = np.zeros((T, N), dtype=np.float32)
    hard_fault_mask = np.zeros((T, N), dtype=bool)
    faults: List[InjectedFault] = []

    eval_ticks = int(duration_s / dt_sim)

    def _pick_visible_sat_at(t_idx: int, min_rem_s: float = 35.0, prefer_high_el: bool = False) -> Optional[int]:
        cands = np.where(
            (orbit.elevation_deg[t_idx] >= min_elev_deg + 2.0)
            & (orbit.rem_vis_sec[t_idx] >= min_rem_s)
            & orbit.gw_reachable[t_idx]
        )[0]
        if len(cands) == 0:
            return None
        if prefer_high_el:
            order = np.argsort(-orbit.elevation_deg[t_idx, cands])
            top = cands[order[: min(3, len(order))]]
            return int(rng.choice(top))
        return int(rng.choice(cands))

    # (a) 区域雨衰 / 城市遮挡：按斜路径长度 1/sin(θ) 放大衰减（低仰角穿越雨区与建筑遮挡更严重）
    n_rain = int(scenario_cfg.get("rain_fade_events", 5))
    rain_starts = np.linspace(60.0, duration_s - 80.0, max(n_rain, 1)) if n_rain > 0 else []
    for base_t in rain_starts:
        start_s = float(np.clip(base_t + rng.uniform(-18.0, 18.0), 35.0, duration_s - 55.0))
        dur_s = float(rng.uniform(30.0, 50.0))
        t0 = int(start_s / dt_sim)
        t1 = min(T, int((start_s + dur_s) / dt_sim))
        # 区域性降雨/云团：天顶基准衰减 zenith_db，沿斜路径按 1/sin(θ) 投影到所有可见星
        zenith_db = float(rng.uniform(1.6, 2.5))
        el_seg = np.clip(orbit.elevation_deg[t0:t1], 10.0, 90.0)
        slant_fade = zenith_db / np.sin(np.radians(el_seg))
        rain_fade_db[t0:t1] = np.maximum(rain_fade_db[t0:t1], slant_fade)
        vis_sats = np.where(orbit.elevation_deg[t0] >= min_elev_deg)[0]
        ref_sat = int(vis_sats[0]) if len(vis_sats) > 0 else 0
        faults.append(
            InjectedFault(
                fault_type="rain_fade",
                sat_idx=ref_sat,
                sat_id=orbit.sat_ids[ref_sat],
                start_s=start_s,
                end_s=start_s + dur_s,
                magnitude=round(zenith_db, 2),
                description=f"区域雨衰 (天顶 {zenith_db:.1f} dB，低仰角斜路径放大)",
            )
        )

    # (b) 突发负载尖峰（随机落在可见卫星上，考验预测校准、负载余量与软触发重规划）
    n_surge = int(scenario_cfg.get("load_surge_events", 6))
    surge_starts = np.linspace(55.0, duration_s - 75.0, max(n_surge, 1)) if n_surge > 0 else []
    for base_t in surge_starts:
        start_s = float(np.clip(base_t + rng.uniform(-18.0, 18.0), 30.0, duration_s - 50.0))
        dur_s = float(rng.uniform(25.0, 45.0))
        t0 = int(start_s / dt_sim)
        t1 = min(T, int((start_s + dur_s) / dt_sim))
        si = _pick_visible_sat_at(t0, min_rem_s=35.0, prefer_high_el=False)
        if si is None:
            continue
        jump = float(rng.uniform(0.32, 0.48))
        true_load[t0:t1, si] = np.clip(true_load[t0:t1, si] + jump, 0.05, 0.99)
        faults.append(
            InjectedFault(
                fault_type="load_surge",
                sat_idx=si,
                sat_id=orbit.sat_ids[si],
                start_s=start_s,
                end_s=start_s + dur_s,
                magnitude=round(jump, 2),
                description=f"负载突发尖峰 (+{jump:.2f}) @ {orbit.sat_ids[si]}",
            )
        )

    # (c) 突发链路/切换硬故障（随机落在可见卫星上，考验确认式切换回退、硬触发重规划与兜底阶梯）
    n_fault = int(scenario_cfg.get("link_fault_events", 5))
    fault_starts = np.linspace(95.0, duration_s - 85.0, max(n_fault, 1)) if n_fault > 0 else []
    for base_t in fault_starts:
        start_s = float(np.clip(base_t + rng.uniform(-22.0, 22.0), 45.0, duration_s - 55.0))
        dur_s = float(rng.uniform(18.0, 32.0))
        t0 = int(start_s / dt_sim)
        t1 = min(T, int((start_s + dur_s) / dt_sim))
        si = _pick_visible_sat_at(t0, min_rem_s=35.0, prefer_high_el=False)
        if si is None:
            continue
        hard_fault_mask[t0:t1, si] = True
        faults.append(
            InjectedFault(
                fault_type="link_fault",
                sat_idx=si,
                sat_id=orbit.sat_ids[si],
                start_s=start_s,
                end_s=start_s + dur_s,
                magnitude=1.0,
                description=f"突发链路/节点故障 @ {orbit.sat_ids[si]}",
            )
        )

    # 4. 计算有效信噪比（低仰角对流层闪烁与多径起伏更大）、端到端时延与物理中断掩码
    clear_snr = _clear_sky_snr_db(orbit.elevation_deg, orbit.slant_range_km)
    el_excess = np.maximum(orbit.elevation_deg - min_elev_deg, 0.0)
    fading_std = 0.75 + 1.5 * np.exp(-el_excess / 4.0)
    fast_fading = (rng.normal(0.0, 1.0, size=(T, N)) * fading_std).astype(np.float32)
    effective_snr_db = (clear_snr - rain_fade_db + fast_fading).astype(np.float32)
    snr_threshold_db = 5.5

    # M/M/1 排队时延近似 + 基础传播/ISL/出口时延
    queue_delay_ms = 9.0 * true_load / np.maximum(1.0 - np.minimum(true_load, 0.96), 0.04)
    e2e_delay_ms = (orbit.prop_delay_ms + queue_delay_ms).astype(np.float32)

    # 物理链路中断：仰角失锁 / 出口不可达 / 突发硬故障 / SNR 低于门限 / 严重过载拥塞崩塌
    overload_drop = (true_load >= 0.945) | (
        (true_load >= 0.90) & (rng.random((T, N)) < (true_load - 0.89) * 4.0)
    )
    link_outage_raw = (
        (orbit.elevation_deg < min_elev_deg)
        | (~orbit.gw_reachable)
        | hard_fault_mask
        | (effective_snr_db < snr_threshold_db)
        | overload_drop
    )

    faults.sort(key=lambda f: f.start_s)

    return NetworkEnvironment(
        orbit=orbit,
        scenario_name=str(scenario_cfg.get("name", "Default")),
        seed=seed,
        min_elev_deg=min_elev_deg,
        true_load=true_load,
        base_mean_load=base_mean_load,
        ar1_phi=phi_plan,
        ar1_sigma_step=sigma_plan,
        rain_fade_db=rain_fade_db,
        hard_fault_mask=hard_fault_mask,
        effective_snr_db=effective_snr_db,
        snr_threshold_db=snr_threshold_db,
        e2e_delay_ms=e2e_delay_ms,
        link_outage_raw=link_outage_raw,
        pred_bias_drift=float(scenario_cfg.get("pred_bias_drift", 0.0)),
        faults=faults,
    )
