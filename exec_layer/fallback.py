"""
四级兜底阶梯模块（exec_layer/fallback.py）
==========================================
对应汇报第 12 页：
- L1 主序列：规划输出的推荐目标星，正常执行；
- L2 备份候选：主目标确认失败或突然不可用时，直接启用随规划同步产出的次优备份星；
- L3 保持当前星：主备均不可靠而当前星仍可见时，暂不切换（接受降质，维持服务不掉线）；
- L4 业务降级：当在用链路严重拥塞或边缘弱信号时，优先保障信令与关键数据，普通业务排队兜底。

追求“降级可控”，每一级的停留步数与代价都会明确写入最终评估报告。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from env.channel_load import NetworkEnvironment


@dataclass
class FallbackDecision:
    level: str              # "L1" | "L2" | "L3" | "L4"
    target_sat: int
    degrade_traffic: bool   # 是否启用 L4 业务降级队列
    reason: str = ""


class FallbackLadder:
    """执行期逐级降级决策器。"""

    def __init__(self, enable_fallback: bool = True) -> None:
        self.enable_fallback = enable_fallback
        self.level_ticks = {"L1": 0, "L2": 0, "L3": 0, "L4": 0}

    def resolve(
        self,
        tick: int,
        curr_sat: int,
        primary_sat: int,
        backup_sat: Optional[int],
        env: NetworkEnvironment,
        primary_failed: bool = False,
    ) -> FallbackDecision:
        """根据当前链路与主/备候选状态，按 L1 → L2 → L3 → L4 阶梯给出执行动作。"""
        if not self.enable_fallback:
            self.level_ticks["L1"] += 1
            return FallbackDecision("L1", primary_sat, False, "未开启兜底阶梯，直行主序列")

        # 1. L1：主序列目标健康可用且未刚发生确认失败
        if not primary_failed and primary_sat >= 0:
            if primary_sat == curr_sat or env.is_sat_healthy_for_confirm(tick, primary_sat):
                degraded = env.is_link_degraded(tick, primary_sat)
                if degraded:
                    self.level_ticks["L4"] += 1
                    return FallbackDecision("L4", primary_sat, True, "主序列链路降质，启用 L4 业务降级")
                self.level_ticks["L1"] += 1
                return FallbackDecision("L1", primary_sat, False, "主序列正常执行")

        # 2. L2：启用随规划同步产出的备份候选
        if backup_sat is not None and backup_sat >= 0 and backup_sat != primary_sat:
            if env.is_sat_healthy_for_confirm(tick, backup_sat):
                self.level_ticks["L2"] += 1
                return FallbackDecision("L2", backup_sat, False, "主序列受阻，立即启用 L2 备份候选")

        # 3. L3：保持当前星（暂不切换，接受降质以维持连续服务）
        curr_visible = (
            curr_sat >= 0
            and env.orbit.elevation_deg[tick, curr_sat] >= env.min_elev_deg
            and env.orbit.gw_reachable[tick, curr_sat]
            and not env.hard_fault_mask[tick, curr_sat]
        )
        if curr_visible:
            if env.is_link_degraded(tick, curr_sat):
                self.level_ticks["L4"] += 1
                return FallbackDecision("L4", curr_sat, True, "保持当前星并启用 L4 业务降级保关键信令")
            self.level_ticks["L3"] += 1
            return FallbackDecision("L3", curr_sat, False, "主备不可靠，L3 保持当前星等待重规划")

        # 4. L4：极端场景——所有优选均受阻，选取当下仰角最高的可达星并强制 L4 业务降级
        vis = np.where((env.orbit.elevation_deg[tick] >= env.min_elev_deg) & env.orbit.gw_reachable[tick])[0]
        if len(vis) > 0:
            # 避开已确认硬故障星
            healthy = [int(s) for s in vis if not env.hard_fault_mask[tick, s]]
            pool = healthy if healthy else [int(s) for s in vis]
            best_emergency = max(pool, key=lambda s: float(env.orbit.elevation_deg[tick, s]))
        else:
            best_emergency = primary_sat if primary_sat >= 0 else curr_sat

        self.level_ticks["L4"] += 1
        return FallbackDecision("L4", best_emergency, True, "极端故障兜底：紧急切至可达星并启用 L4 业务降级")
