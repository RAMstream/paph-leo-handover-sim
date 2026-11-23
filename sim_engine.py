"""
主仿真引擎（sim_engine.py）
===========================
在统一物理环境与微步长 δt 上运行任意策略（基线 / PAPH / 消融变体），
精确统计汇报第 14、15 页定义的所有核心服务体验指标与实时性预算指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from baselines.policies import BasePolicy
from env.channel_load import NetworkEnvironment
from env.traffic import DualPriorityTrafficModel


@dataclass
class SimulationResult:
    """单次策略仿真运行后的汇总指标与逐时刻明细轨迹。"""

    policy_name: str
    scenario_name: str
    duration_s: float
    # 核心服务体验指标（第 15 页）
    outage_ratio_pct: float           # 实际累计中断占比 O (%)
    total_outage_s: float             # 累计中断总时长 (s)
    max_single_outage_s: float        # 最长单次连续中断时长 (s)
    handover_count: int               # 成功切换次数
    confirm_failures: int             # 确认式切换拦下的失败次数（回退保护）
    mean_e2e_delay_ms: float          # 连通时刻平均端到端时延 D (ms)
    p95_e2e_delay_ms: float           # P95 端到端时延 (ms)
    # 业务连续性与兜底代价（第 12 页）
    critical_loss_pct: float          # 关键信令/数据丢包率 (%)
    normal_loss_pct: float            # 普通业务丢包率 (%)
    fallback_dist_pct: Dict[str, float] = field(default_factory=dict)
    # 实时性预算统计（第 14 页）
    replan_count: int = 0
    event_trigger_count: int = 0
    mean_plan_cpu_ms: float = 0.0     # 单次规划纯计算耗时 (更新+筛选+DP)
    max_plan_cpu_ms: float = 0.0      # 最坏单次规划耗时
    mean_stage_ms: Dict[str, float] = field(default_factory=dict)
    # 逐微步明细轨迹（用于绘制时间轴回放图）
    trace: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, object]:
        return {
            "Strategy": self.policy_name,
            "OutageRatio(%)": round(self.outage_ratio_pct, 2),
            "MaxSingleOutage(s)": round(self.max_single_outage_s, 2),
            "Handovers": self.handover_count,
            "ConfirmRollbacks": self.confirm_failures,
            "MeanDelay(ms)": round(self.mean_e2e_delay_ms, 2),
            "P95Delay(ms)": round(self.p95_e2e_delay_ms, 2),
            "CritLoss(%)": round(self.critical_loss_pct, 2),
            "EventTriggers": self.event_trigger_count,
            "MeanPlanCPU(ms)": round(self.mean_plan_cpu_ms, 3),
            "MaxPlanCPU(ms)": round(self.max_plan_cpu_ms, 3),
        }


def run_simulation(policy: BasePolicy, env: NetworkEnvironment, cfg: Dict) -> SimulationResult:
    """执行完整仿真并返回统计结果与明细轨迹。"""
    dt_sim = env.orbit.dt_sim
    duration_s = float(cfg["simulation"]["duration_s"])
    n_ticks = int(round(duration_s / dt_sim))

    traffic = DualPriorityTrafficModel(max_queue_packets=90)
    curr_sat = policy.initial_sat()
    prev_outage = False

    # 逐微步记录数组
    times_arr = env.orbit.times_s[:n_ticks].copy()
    sat_arr = np.zeros(n_ticks, dtype=np.int32)
    outage_sec_arr = np.zeros(n_ticks, dtype=np.float32)
    delay_arr = np.zeros(n_ticks, dtype=np.float32)
    load_arr = np.zeros(n_ticks, dtype=np.float32)
    elev_arr = np.zeros(n_ticks, dtype=np.float32)
    fb_arr = np.empty(n_ticks, dtype=object)
    trig_arr = np.empty(n_ticks, dtype=object)

    # 阶段耗时记录（仅记录发生规划的微步）
    update_ms_list: List[float] = []
    filter_ms_list: List[float] = []
    dp_ms_list: List[float] = []
    sig_ms_list: List[float] = []
    total_plan_ms_list: List[float] = []

    curr_outage_run_s = 0.0
    max_outage_run_s = 0.0
    connected_delays: List[float] = []

    for tick in range(n_ticks):
        res = policy.step(tick, curr_sat, prev_outage)
        curr_sat = res.active_sat

        # 物理链路实际状态
        raw_outage = bool(env.link_outage_raw[tick, curr_sat])
        ho_hit_s = float(res.ho_outcome.ho_interruption_s)

        if raw_outage:
            tick_out_s = dt_sim
        else:
            tick_out_s = min(dt_sim, ho_hit_s)

        # 连续中断时长累计（注：极短 MBB 切换 35 ms 单独计入中断时长，连续整拍断网累加）
        if tick_out_s >= dt_sim - 1e-6:
            curr_outage_run_s += tick_out_s
            max_outage_run_s = max(max_outage_run_s, curr_outage_run_s)
        elif tick_out_s > 0.0:
            max_outage_run_s = max(max_outage_run_s, curr_outage_run_s + tick_out_s)
            curr_outage_run_s = 0.0
        else:
            curr_outage_run_s = 0.0

        link_up = not raw_outage and (ho_hit_s < dt_sim * 0.8)
        sat_load = float(env.true_load[tick, curr_sat])
        sat_snr = float(env.effective_snr_db[tick, curr_sat])
        sat_delay = float(env.e2e_delay_ms[tick, curr_sat])

        # 业务流推进（L4 业务降级保护关键信令）
        traffic.step(
            link_up=link_up,
            degraded_l4=res.degrade_traffic,
            sat_load=sat_load,
            snr_db=sat_snr,
            snr_th_db=env.snr_threshold_db,
        )

        if link_up:
            connected_delays.append(sat_delay)

        prev_outage = raw_outage

        sat_arr[tick] = curr_sat
        outage_sec_arr[tick] = tick_out_s
        delay_arr[tick] = sat_delay if link_up else np.nan
        load_arr[tick] = sat_load
        elev_arr[tick] = float(env.orbit.elevation_deg[tick, curr_sat])
        fb_arr[tick] = res.fallback_level
        trig_arr[tick] = res.trigger_type

        if res.trigger_type != "NONE":
            update_ms_list.append(res.update_forecast_ms)
            filter_ms_list.append(res.filter_ms)
            dp_ms_list.append(res.dp_ms)
            sig_ms_list.append(res.signaling_ms)
            total_plan_ms_list.append(res.update_forecast_ms + res.filter_ms + res.dp_ms)

    total_outage_s = float(np.sum(outage_sec_arr))
    outage_ratio_pct = 100.0 * total_outage_s / duration_s
    mean_delay = float(np.mean(connected_delays)) if connected_delays else 999.0
    p95_delay = float(np.percentile(connected_delays, 95)) if connected_delays else 999.0

    # 四级兜底分布占比
    fb_tot = max(sum(policy.fallback.level_ticks.values()), 1)
    fb_dist = {k: round(100.0 * v / fb_tot, 2) for k, v in policy.fallback.level_ticks.items()}

    mean_stages = {
        "update_forecast_ms": float(np.mean(update_ms_list)) if update_ms_list else 0.0,
        "filter_ms": float(np.mean(filter_ms_list)) if filter_ms_list else 0.0,
        "dp_ms": float(np.mean(dp_ms_list)) if dp_ms_list else 0.0,
        "signaling_ms": float(np.mean(sig_ms_list)) if sig_ms_list else 0.0,
    }

    return SimulationResult(
        policy_name=policy.name,
        scenario_name=env.scenario_name,
        duration_s=duration_s,
        outage_ratio_pct=outage_ratio_pct,
        total_outage_s=total_outage_s,
        max_single_outage_s=max_outage_run_s,
        handover_count=policy.fsm.total_handovers,
        confirm_failures=policy.fsm.confirm_failures,
        mean_e2e_delay_ms=mean_delay,
        p95_e2e_delay_ms=p95_delay,
        critical_loss_pct=100.0 * traffic.stats.critical_loss_rate,
        normal_loss_pct=100.0 * traffic.stats.normal_loss_rate,
        fallback_dist_pct=fb_dist,
        replan_count=policy.replan_count,
        event_trigger_count=policy.event_trigger_count,
        mean_plan_cpu_ms=float(np.mean(total_plan_ms_list)) if total_plan_ms_list else 0.0,
        max_plan_cpu_ms=float(np.max(total_plan_ms_list)) if total_plan_ms_list else 0.0,
        mean_stage_ms=mean_stages,
        trace={
            "times_s": times_arr,
            "sat_idx": sat_arr,
            "outage_s": outage_sec_arr,
            "delay_ms": delay_arr,
            "load": load_arr,
            "elev_deg": elev_arr,
            "fallback": fb_arr,
            "trigger": trig_arr,
        },
    )
