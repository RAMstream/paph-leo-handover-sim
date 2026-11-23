"""
统一策略接口：对照基线与 PAPH（含消融实验开关）
================================================
对应汇报第 15 页：
对照基线（3 个）：
  1. StrongestSignalPolicy：最强信号（每 Δt 选仰角/信号最高星，传统先断后连）
  2. MaxVisiblePolicy：最长可见（当前星快离开时，选剩余可见时间最长的星）
  3. WeightedMyopicPolicy：接入加权（仅基于当下瞬时快照按 AHP 权重选星，无未来窗口规划与执行期保障）
本方案与消融变体：
  4. PAPHPolicy：通过开关组合一键生成
     - PAPH (完整方案)
     - 消融 A：去余量 (use_margin=False)
     - 消融 B：去确认式切换 (use_mbb=False)
     - 消融 C：去触发重规划 (use_event_trigger=False)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import time
import numpy as np

from core.monitor import ExecutionMonitor
from core.planner import MarginDPPlanner, PlanResult, compute_ahp_weights
from core.predictor import RollingPredictor
from env.channel_load import NetworkEnvironment
from exec_layer.fallback import FallbackLadder
from exec_layer.handover_fsm import HandoverFSM, HandoverStepOutcome


@dataclass
class PolicyStepResult:
    """单个微步长策略执行后的完整输出，供仿真引擎统计与画图。"""

    active_sat: int
    ho_outcome: HandoverStepOutcome
    fallback_level: str
    degrade_traffic: bool
    trigger_type: str
    update_forecast_ms: float = 0.0
    filter_ms: float = 0.0
    dp_ms: float = 0.0
    signaling_ms: float = 0.0


class BasePolicy:
    """所有切换策略的统一抽象基类。"""

    def __init__(self, name: str, env: NetworkEnvironment, cfg: Dict) -> None:
        self.name = name
        self.env = env
        self.cfg = cfg
        self.dt_sim = env.orbit.dt_sim
        self.dt_plan = float(cfg["simulation"]["dt_plan"])
        self.plan_interval_ticks = max(1, int(round(self.dt_plan / self.dt_sim)))
        self.min_elev_deg = float(cfg["simulation"]["min_elev_deg"])
        self.fsm = HandoverFSM(
            dt_sim=self.dt_sim,
            confirm_delay_s=float(cfg["paph"]["confirm_delay_s"]),
            mbb_hit_s=float(cfg["paph"]["mbb_hit_s"]),
            bbm_hit_s=float(cfg["paph"]["bbm_hit_s"]),
            use_mbb=False,
        )
        self.fallback = FallbackLadder(enable_fallback=False)
        self.replan_count = 0
        self.event_trigger_count = 0

    def initial_sat(self) -> int:
        vis = np.where(
            (self.env.orbit.elevation_deg[0] >= self.min_elev_deg) & self.env.orbit.gw_reachable[0]
        )[0]
        if len(vis) == 0:
            return int(np.argmax(self.env.orbit.elevation_deg[0]))
        return int(vis[np.argmax(self.env.orbit.elevation_deg[0, vis])])

    def step(self, tick: int, curr_sat: int, prev_link_outage: bool) -> PolicyStepResult:
        raise NotImplementedError


class StrongestSignalPolicy(BasePolicy):
    """基线 1：最强信号（每 Δt 选当前仰角/晴空信号最强的可见卫星，未考虑下游负载与未来离开）。"""

    def __init__(self, env: NetworkEnvironment, cfg: Dict) -> None:
        super().__init__("Strongest-Signal (最强信号)", env, cfg)

    def step(self, tick: int, curr_sat: int, prev_link_outage: bool) -> PolicyStepResult:
        t0 = time.perf_counter()
        trigger = "NONE"
        target = curr_sat
        if tick % self.plan_interval_ticks == 0:
            trigger = "PERIODIC"
            self.replan_count += 1
            vis = np.where(
                (self.env.orbit.elevation_deg[tick] >= self.min_elev_deg) & self.env.orbit.gw_reachable[tick]
            )[0]
            if len(vis) > 0:
                best = int(vis[np.argmax(self.env.orbit.elevation_deg[tick, vis])])
                # 小迟滞（1.5°）防止同仰角抖动
                if (
                    curr_sat not in vis
                    or self.env.orbit.elevation_deg[tick, best] > self.env.orbit.elevation_deg[tick, curr_sat] + 1.5
                ):
                    target = best
        comp_ms = (time.perf_counter() - t0) * 1000.0
        ho = self.fsm.request_handover(curr_sat, target, tick, self.env)
        return PolicyStepResult(
            active_sat=ho.active_sat,
            ho_outcome=ho,
            fallback_level="L1",
            degrade_traffic=False,
            trigger_type=trigger,
            update_forecast_ms=comp_ms,
            signaling_ms=ho.signaling_ms,
        )


class MaxVisiblePolicy(BasePolicy):
    """基线 2：最长可见时间（减少切换次数，但不规避高负载、雨衰和突发故障）。"""

    def __init__(self, env: NetworkEnvironment, cfg: Dict) -> None:
        super().__init__("Max-Visible (最长可见)", env, cfg)

    def step(self, tick: int, curr_sat: int, prev_link_outage: bool) -> PolicyStepResult:
        t0 = time.perf_counter()
        trigger = "NONE"
        target = curr_sat
        if tick % self.plan_interval_ticks == 0:
            trigger = "PERIODIC"
            self.replan_count += 1
            curr_ok = (
                self.env.orbit.elevation_deg[tick, curr_sat] >= self.min_elev_deg
                and self.env.orbit.gw_reachable[tick, curr_sat]
                and self.env.orbit.rem_vis_sec[tick, curr_sat] > self.dt_plan
            )
            if not curr_ok:
                vis = np.where(
                    (self.env.orbit.elevation_deg[tick] >= self.min_elev_deg) & self.env.orbit.gw_reachable[tick]
                )[0]
                if len(vis) > 0:
                    target = int(vis[np.argmax(self.env.orbit.rem_vis_sec[tick, vis])])
        comp_ms = (time.perf_counter() - t0) * 1000.0
        ho = self.fsm.request_handover(curr_sat, target, tick, self.env)
        return PolicyStepResult(
            active_sat=ho.active_sat,
            ho_outcome=ho,
            fallback_level="L1",
            degrade_traffic=False,
            trigger_type=trigger,
            update_forecast_ms=comp_ms,
            signaling_ms=ho.signaling_ms,
        )


class WeightedMyopicPolicy(BasePolicy):
    """基线 3：接入加权（仅依据当下观测按 AHP 权重贪心选星，无多步规划、无余量、无确认式切换）。"""

    def __init__(self, env: NetworkEnvironment, cfg: Dict) -> None:
        super().__init__("Weighted-Myopic (接入加权)", env, cfg)
        w, _ = compute_ahp_weights(cfg["paph"]["ahp_pairwise"])
        self.alpha, self.beta, self.gamma, self.delta = [float(x) for x in w]

    def step(self, tick: int, curr_sat: int, prev_link_outage: bool) -> PolicyStepResult:
        t0 = time.perf_counter()
        trigger = "NONE"
        target = curr_sat
        if tick % self.plan_interval_ticks == 0:
            trigger = "PERIODIC"
            self.replan_count += 1
            vis = np.where(
                (self.env.orbit.elevation_deg[tick] >= self.min_elev_deg) & self.env.orbit.gw_reachable[tick]
            )[0]
            if len(vis) > 0:
                best_s = curr_sat
                best_c = float("inf")
                for s in vis:
                    si = int(s)
                    d_norm = min(float(self.env.e2e_delay_ms[tick, si]) / 90.0, 2.5)
                    l_val = float(self.env.true_load[tick, si])
                    # 当下瞬时风险只看当前是否超低仰角或严重拥塞
                    o_now = 1.0 if self.env.link_outage_raw[tick, si] else (0.15 if l_val > 0.85 else 0.02)
                    sw = 0.0 if si == curr_sat else self.beta
                    cost = self.alpha * d_norm + sw + self.gamma * l_val + self.delta * o_now
                    if cost < best_c:
                        best_c = cost
                        best_s = si
                target = best_s
        comp_ms = (time.perf_counter() - t0) * 1000.0
        ho = self.fsm.request_handover(curr_sat, target, tick, self.env)
        return PolicyStepResult(
            active_sat=ho.active_sat,
            ho_outcome=ho,
            fallback_level="L1",
            degrade_traffic=False,
            trigger_type=trigger,
            update_forecast_ms=comp_ms,
            signaling_ms=ho.signaling_ms,
        )


class PAPHPolicy(BasePolicy):
    """
    PAPH 完整执行期保障方案（支持精细消融开关）：
    - 机制一：预测与滚动校准 (RollingPredictor)
    - 机制二：带余量与机会约束的 DP 规划 (MarginDPPlanner, 开关 use_margin)
    - 机制三：Make-before-break 确认式切换 (HandoverFSM, 开关 use_mbb)
    - 机制四：在线监控与软/硬触发增量重规划 (ExecutionMonitor, 开关 use_event_trigger)
    - 兜底阶梯：L1 主序列 → L2 备份候选 → L3 保持当前星 → L4 业务降级 (FallbackLadder)
    """

    def __init__(
        self,
        env: NetworkEnvironment,
        cfg: Dict,
        name: str = "PAPH (本方案)",
        use_margin: bool = True,
        use_mbb: bool = True,
        use_event_trigger: bool = True,
        use_fallback: bool = True,
        use_calibration: bool = True,
        horizon_override: Optional[int] = None,
        k_override: Optional[int] = None,
        epsilon_override: Optional[float] = None,
    ) -> None:
        super().__init__(name, env, cfg)
        sim_cfg = cfg["simulation"]
        paph_cfg = cfg["paph"]

        self.H = int(horizon_override if horizon_override is not None else sim_cfg["horizon_H"])
        self.K = int(k_override if k_override is not None else sim_cfg["max_candidates_K"])
        eps = float(epsilon_override if epsilon_override is not None else paph_cfg["chance_epsilon"])

        self.predictor = RollingPredictor(
            env=env,
            dt_plan=self.dt_plan,
            ewma_alpha=float(paph_cfg["ewma_alpha"]),
            enable_calibration=use_calibration,
        )
        self.planner = MarginDPPlanner(
            min_elev_deg=self.min_elev_deg,
            max_candidates_K=self.K,
            ahp_pairwise=paph_cfg["ahp_pairwise"],
            elev_margin_deg=float(paph_cfg["elev_margin_deg"]),
            min_rem_vis_s=float(paph_cfg["min_rem_vis_s"]),
            risk_kappa_load=float(paph_cfg["risk_kappa_load"]),
            risk_kappa_outage=float(paph_cfg["risk_kappa_outage"]),
            chance_epsilon=eps,
            use_margin=use_margin,
        )
        self.monitor = ExecutionMonitor(
            dt_sim=self.dt_sim,
            dt_plan=self.dt_plan,
            soft_trigger_sigma=float(paph_cfg["soft_trigger_sigma"]),
            soft_trigger_persist=int(paph_cfg["soft_trigger_persist"]),
            replan_cooldown_s=float(paph_cfg["replan_cooldown_s"]),
            enable_event_trigger=use_event_trigger,
        )
        self.fsm = HandoverFSM(
            dt_sim=self.dt_sim,
            confirm_delay_s=float(paph_cfg["confirm_delay_s"]),
            mbb_hit_s=float(paph_cfg["mbb_hit_s"]),
            bbm_hit_s=float(paph_cfg["bbm_hit_s"]),
            use_mbb=use_mbb,
        )
        self.fallback = FallbackLadder(enable_fallback=use_fallback)

        self.last_plan: Optional[PlanResult] = None
        self.recent_failed_sats: Dict[int, int] = {}  # sat_idx -> 解除隔离 tick

    def initial_sat(self) -> int:
        # 初始接入即运行一次带余量规划，避免冷启动落在边缘差星上
        self.predictor.observe_and_calibrate(0)
        fw = self.predictor.forecast(0, self.H)
        plan = self.planner.solve(fw, curr_sat=-1)
        self.last_plan = plan
        return plan.primary_sat

    def step(self, tick: int, curr_sat: int, prev_link_outage: bool) -> PolicyStepResult:
        # 清理已过期的失败隔离记录
        self.recent_failed_sats = {s: exp for s, exp in self.recent_failed_sats.items() if exp > tick}

        # 1. 若有正在进行的 Make-before-break 确认流程，先推进一拍
        pending_ho = self.fsm.tick_pending(curr_sat, tick, self.env)
        active_sat = pending_ho.active_sat
        ho_confirm_failed = pending_ho.confirm_failed
        failed_sat = pending_ho.failed_target_sat

        if ho_confirm_failed and failed_sat is not None:
            # 确认失败的目标星隔离 20 秒，避免反复撞墙
            self.recent_failed_sats[failed_sat] = tick + int(round(20.0 / self.dt_sim))

        # 2. 机制一：观测当前状态并滚动校准偏差模型
        t_u0 = time.perf_counter()
        obs_info = self.predictor.observe_and_calibrate(tick, serving_sat=active_sat, failed_sat=failed_sat)
        curr_outage_now = bool(self.env.link_outage_raw[tick, active_sat])
        curr_elev = float(self.env.orbit.elevation_deg[tick, active_sat])

        # 3. 机制四：监控并判定周期/软/硬触发
        trig = self.monitor.evaluate(
            tick=tick,
            obs_info=obs_info,
            link_outage=curr_outage_now,
            ho_confirm_failed=ho_confirm_failed,
            curr_elev_deg=curr_elev,
            min_elev_deg=self.min_elev_deg,
        )

        filter_ms = 0.0
        dp_ms = 0.0
        primary_failed_flag = ho_confirm_failed

        if trig.should_replan:
            self.replan_count += 1
            if trig.trigger_type != "PERIODIC":
                self.event_trigger_count += 1
            fw = self.predictor.forecast(tick, self.H)
            update_forecast_ms = (time.perf_counter() - t_u0) * 1000.0

            exclude = set(self.recent_failed_sats.keys())
            if curr_outage_now:
                exclude.add(active_sat)

            inc_layers = min(6, self.H) if trig.use_incremental else None
            plan = self.planner.solve(
                fw,
                curr_sat=active_sat,
                exclude_first_step=exclude,
                incremental_layers=inc_layers,
            )
            self.last_plan = plan
            filter_ms = plan.filter_time_ms
            dp_ms = plan.dp_time_ms
            primary_failed_flag = False  # 刚重新规划完毕，启用新主序列
        else:
            update_forecast_ms = (time.perf_counter() - t_u0) * 1000.0

        # 4. 兜底阶梯（L1 -> L2 -> L3 -> L4）决定本微步最终目标与降级状态
        if self.last_plan is not None:
            p_sat = self.last_plan.primary_sat
            b_sat = self.last_plan.backup_sat
        else:
            p_sat = active_sat
            b_sat = None

        # 如果既未触发新规划、也无确认失败，且当前正在准备 MBB 或稳态服务，检查当前星降质状态即可
        if not trig.should_replan and not ho_confirm_failed:
            degraded_now = self.fallback.enable_fallback and self.env.is_link_degraded(tick, active_sat)
            lvl = "L4" if degraded_now else "L1"
            self.fallback.level_ticks[lvl] += 1
            return PolicyStepResult(
                active_sat=active_sat,
                ho_outcome=pending_ho,
                fallback_level=lvl,
                degrade_traffic=degraded_now,
                trigger_type=trig.trigger_type,
                update_forecast_ms=update_forecast_ms,
                filter_ms=filter_ms,
                dp_ms=dp_ms,
                signaling_ms=pending_ho.signaling_ms,
            )

        fb = self.fallback.resolve(
            tick=tick,
            curr_sat=active_sat,
            primary_sat=p_sat,
            backup_sat=b_sat,
            env=self.env,
            primary_failed=primary_failed_flag,
        )

        # 5. 机制三：执行层发起切换（Make-before-break 或对照 BBM）
        if fb.target_sat != active_sat:
            ho_out = self.fsm.request_handover(active_sat, fb.target_sat, tick, self.env)
            # 合并同一拍可能存在的确认失败信息与新发起切换信令
            ho_out.confirm_failed = ho_out.confirm_failed or pending_ho.confirm_failed
            ho_out.rolled_back = ho_out.rolled_back or pending_ho.rolled_back
            ho_out.ho_interruption_s = max(ho_out.ho_interruption_s, pending_ho.ho_interruption_s)
            ho_out.signaling_ms += pending_ho.signaling_ms
        else:
            ho_out = pending_ho

        return PolicyStepResult(
            active_sat=ho_out.active_sat,
            ho_outcome=ho_out,
            fallback_level=fb.level,
            degrade_traffic=fb.degrade_traffic,
            trigger_type=trig.trigger_type,
            update_forecast_ms=update_forecast_ms,
            filter_ms=filter_ms,
            dp_ms=dp_ms,
            signaling_ms=ho_out.signaling_ms,
        )
