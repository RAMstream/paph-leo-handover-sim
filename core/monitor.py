"""
机制四 · 在线监控与触发式重规划判定模块（core/monitor.py）
==========================================================
对应汇报第 7、11 页：
- 监控已有测量（实测与预测偏差、链路质量、切换执行结果），不新增额外信令。
- 三类触发源：
  1. PERIODIC：每 Δt 周期滚动更新；
  2. SOFT_DEVIATION（软触发）：实测负载/质量持续偏离预测区间超限；
  3. HARD_FAULT / HARD_HO_FAIL（硬触发）：切换确认失败或当前链路异常失步，立即触发增量重规划。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class TriggerDecision:
    should_replan: bool
    trigger_type: str      # "NONE" | "PERIODIC" | "SOFT_DEVIATION" | "HARD_FAULT" | "HARD_HO_FAIL"
    use_incremental: bool  # 突发事件触发时使用局部增量更新，守住实时预算
    reason: str = ""


class ExecutionMonitor:
    """运行在每个物理微步长 δt 上的执行期监控器。"""

    def __init__(
        self,
        dt_sim: float,
        dt_plan: float,
        soft_trigger_sigma: float = 1.8,
        soft_trigger_persist: int = 3,
        replan_cooldown_s: float = 2.0,
        enable_event_trigger: bool = True,
    ) -> None:
        self.dt_sim = dt_sim
        self.plan_interval_ticks = max(1, int(round(dt_plan / dt_sim)))
        self.soft_trigger_sigma = soft_trigger_sigma
        self.soft_trigger_persist = soft_trigger_persist
        self.cooldown_ticks = max(1, int(round(replan_cooldown_s / dt_sim)))
        self.enable_event_trigger = enable_event_trigger

        self.deviation_streak = 0
        self.last_replan_tick = -9999

    def evaluate(
        self,
        tick: int,
        obs_info: Dict[str, float],
        link_outage: bool,
        ho_confirm_failed: bool,
        curr_elev_deg: float,
        min_elev_deg: float,
    ) -> TriggerDecision:
        """在当前微步长判定是否需要立即重规划。"""
        # 1. 硬触发（最高优先级）：切换确认失败
        if ho_confirm_failed:
            if self.enable_event_trigger:
                self.deviation_streak = 0
                self.last_replan_tick = tick
                return TriggerDecision(True, "HARD_HO_FAIL", True, "切换确认失败，立即作废旧序列并重规划")

        # 2. 硬触发：当前链路中断失步或逼近仰角失锁
        in_cooldown = (tick - self.last_replan_tick) < self.cooldown_ticks
        if (link_outage or curr_elev_deg < min_elev_deg + 1.0) and self.enable_event_trigger and not in_cooldown:
            self.deviation_streak = 0
            self.last_replan_tick = tick
            return TriggerDecision(True, "HARD_FAULT", True, "当前链路异常/逼近失锁，硬触发重规划")

        # 3. 周期滚动触发（每 Δt 触发全窗口滚动规划）
        if tick % self.plan_interval_ticks == 0:
            self.deviation_streak = 0
            self.last_replan_tick = tick
            return TriggerDecision(True, "PERIODIC", False, "周期 Δt 滚动规划")

        # 4. 软触发：实测负载持续高于预测区间上限（偏差超限）
        if self.enable_event_trigger and not in_cooldown:
            res = obs_info.get("serving_residual", 0.0)
            sig = max(obs_info.get("serving_sigma", 0.06), 0.04)
            load = obs_info.get("serving_load", 0.0)
            if (res > self.soft_trigger_sigma * sig and load > 0.72) or load >= 0.88:
                self.deviation_streak += 1
            else:
                self.deviation_streak = 0

            if self.deviation_streak >= self.soft_trigger_persist:
                self.deviation_streak = 0
                self.last_replan_tick = tick
                return TriggerDecision(
                    True,
                    "SOFT_DEVIATION",
                    True,
                    f"实测负载持续偏离预测 (ΔL={res:+.2f}, L={load:.2f})，软触发增量重规划",
                )

        return TriggerDecision(False, "NONE", False, "")
