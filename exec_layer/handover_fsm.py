"""
机制三 · 确认式切换状态机（exec_layer/handover_fsm.py）
=======================================================
对应汇报第 10 页（参考 3GPP TS 38.300 Rel-18 NTN RACH-less / CHO 思想）：
- Make-before-break（use_mbb=True）：
  1. 当前链路持续服务（绝不提前放手）；
  2. 目标链路并行建立与确认（耗时 confirm_delay_s）；
  3. 确认成功 → 平滑接管，仅产生极短切换中断（mbb_hit_s ≈ 35 ms）；
  4. 确认失败 → 回退当前卫星继续服务（不白白掉线！），向上层报告失败以触发 L2/L3 兜底与重规划。
- Break-before-make 对照模式（use_mbb=False）：
  直接断开旧链路、切换至目标星，每次切换固定产生 bbm_hit_s 中断；
  若目标星恰逢故障/过载，则直接掉入中断直至后续恢复。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from env.channel_load import NetworkEnvironment


@dataclass
class HandoverStepOutcome:
    """单个微步长内切换状态机的执行结果。"""

    active_sat: int
    handover_completed: bool = False
    confirm_failed: bool = False
    failed_target_sat: Optional[int] = None
    rolled_back: bool = False
    ho_interruption_s: float = 0.0  # 本微步内因切换本身造成的物理中断时长 (秒)
    signaling_ms: float = 0.0       # 本微步切换信令/确认流程开销估算 (ms)


class HandoverFSM:
    """管理星地链路切换生命周期的状态机。"""

    def __init__(
        self,
        dt_sim: float,
        confirm_delay_s: float = 0.5,
        mbb_hit_s: float = 0.035,
        bbm_hit_s: float = 0.5,
        use_mbb: bool = True,
    ) -> None:
        self.dt_sim = dt_sim
        self.confirm_ticks = max(1, int(round(confirm_delay_s / dt_sim)))
        self.mbb_hit_s = mbb_hit_s
        self.bbm_hit_s = bbm_hit_s
        self.use_mbb = use_mbb

        # 当前状态
        self.state: str = "IDLE"          # "IDLE" | "PREPARING_MBB"
        self.pending_target: Optional[int] = None
        self.ticks_remaining: int = 0

        # 统计计数
        self.total_handovers: int = 0
        self.confirm_failures: int = 0
        self.rollbacks: int = 0

    def request_handover(self, curr_sat: int, target_sat: int, tick: int, env: NetworkEnvironment) -> HandoverStepOutcome:
        """
        收到上层（规划/兜底）的切换指令。
        若目标星等于当前星，直接保持服务；否则按 MBB 或 BBM 执行。
        """
        if target_sat == curr_sat or target_sat < 0:
            self.state = "IDLE"
            self.pending_target = None
            return HandoverStepOutcome(active_sat=curr_sat)

        if not self.use_mbb:
            # 传统先断后连（Break-Before-Make）：直接断旧连新，承受固定切换中断
            self.total_handovers += 1
            self.state = "IDLE"
            self.pending_target = None
            return HandoverStepOutcome(
                active_sat=target_sat,
                handover_completed=True,
                ho_interruption_s=min(self.bbm_hit_s, self.dt_sim),
                signaling_ms=42.0,
            )

        # Make-before-break：若已在准备同一目标则继续等待；否则启动新目标准备
        if self.state == "PREPARING_MBB" and self.pending_target == target_sat:
            return self.tick_pending(curr_sat, tick, env)

        self.state = "PREPARING_MBB"
        self.pending_target = target_sat
        self.ticks_remaining = self.confirm_ticks
        # 启动确认准备的这一微步，旧链路继续服务（0 中断）
        return HandoverStepOutcome(active_sat=curr_sat, signaling_ms=18.0)

    def tick_pending(self, curr_sat: int, tick: int, env: NetworkEnvironment) -> HandoverStepOutcome:
        """推进正在进行中的 Make-before-break 双链路确认流程。"""
        if self.state != "PREPARING_MBB" or self.pending_target is None:
            return HandoverStepOutcome(active_sat=curr_sat)

        self.ticks_remaining -= 1
        if self.ticks_remaining > 0:
            return HandoverStepOutcome(active_sat=curr_sat, signaling_ms=5.0)

        # 到达确认判定点：验证目标链路是否健康可接管
        target = self.pending_target
        self.state = "IDLE"
        self.pending_target = None

        if env.is_sat_healthy_for_confirm(tick, target):
            self.total_handovers += 1
            return HandoverStepOutcome(
                active_sat=target,
                handover_completed=True,
                ho_interruption_s=self.mbb_hit_s,
                signaling_ms=24.0,
            )

        # 确认失败：绝不盲切，回退至当前服务星继续工作！
        self.confirm_failures += 1
        self.rollbacks += 1
        return HandoverStepOutcome(
            active_sat=curr_sat,
            confirm_failed=True,
            failed_target_sat=target,
            rolled_back=True,
            ho_interruption_s=0.0,
            signaling_ms=28.0,
        )
