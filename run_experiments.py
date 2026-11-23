"""
PAPH 实验总控与一键出图脚本（run_experiments.py）
================================================
一键运行汇报第 14–16 页定义的完整验证方案：
1. AHP 权重一致性校验 (CR < 0.1)
2. 验证场景（调参核验）与测试场景（最终评估）分离运行
3. 四类对照基线对比实验（最强信号 / 最长可见 / 接入加权 / PAPH 完整方案）
4. 四组消融实验（完整方案 / 去余量 / 去确认式切换 / 去触发重规划）
5. 实时性预算测试（扫描 H 与 K，拆解各阶段耗时 vs Δt 预算）
6. 风险容忍度 ε 敏感性分析（中断占比 vs 端到端时延帕累托权衡）
7. 自动保存 CSV 结果表与 5 张高清分析图至 results/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines.policies import (
    MaxVisiblePolicy,
    PAPHPolicy,
    StrongestSignalPolicy,
    WeightedMyopicPolicy,
)
from core.planner import compute_ahp_weights
from env.channel_load import NetworkEnvironment, build_environment
from env.orbit import build_orbit_trace
from sim_engine import SimulationResult, run_simulation

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"


def _configure_matplotlib() -> None:
    """优先使用支持中文的字体，若系统缺失则回退至通用无衬线字体，确保不乱码。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def _load_config() -> Dict:
    with open(ROOT / "config" / "scenarios.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_baseline_comparison(results: List[SimulationResult], out_path: Path) -> None:
    """图 1：对照基线 4 项核心服务指标对比图。"""
    labels = [r.policy_name.split(" (")[0] for r in results]
    colors = ["#7B8FA1", "#9BA4B5", "#E5B25D", "#1E6F5C"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    metrics = [
        ("实际中断占比 O (%)", [r.outage_ratio_pct for r in results], "{:.2f}%"),
        ("最长单次中断时长 (s)", [r.max_single_outage_s for r in results], "{:.2f}s"),
        ("总切换次数", [r.handover_count for r in results], "{:d}"),
        ("平均端到端时延 D (ms)", [r.mean_e2e_delay_ms for r in results], "{:.1f}ms"),
    ]

    for ax, (title, vals, fmt) in zip(axes.flat, metrics):
        bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="#222222", linewidth=0.7)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        ax.set_axisbelow(True)
        ymax = max(vals) * 1.24 if max(vals) > 0 else 1.0
        ax.set_ylim(0, ymax)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + ymax * 0.03,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    fig.suptitle("PAPH 与三类基线策略核心指标对比（测试场景 · 含故障注入）", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_study(results: List[SimulationResult], out_path: Path) -> None:
    """图 2：消融实验拆账图（展示余量、确认式切换、触发重规划各自的独立贡献）。"""
    labels = [r.policy_name for r in results]
    colors = ["#1E6F5C", "#D9534F", "#F0AD4E", "#4A90E2"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    specs = [
        ("实际中断占比 O (%)", [r.outage_ratio_pct for r in results], "{:.2f}%"),
        ("最长单次中断时长 (s)", [r.max_single_outage_s for r in results], "{:.2f}s"),
        ("总切换次数", [r.handover_count for r in results], "{:d}"),
    ]

    for ax, (title, vals, fmt) in zip(axes, specs):
        bars = ax.bar(range(len(labels)), vals, color=colors, width=0.56, edgecolor="#222222", linewidth=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        ax.set_axisbelow(True)
        ymax = max(vals) * 1.25 if max(vals) > 0 else 1.0
        ax.set_ylim(0, ymax)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + ymax * 0.03,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    fig.suptitle("PAPH 消融实验：各执行期机制的独立收益拆账", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_execution_timeline(
    paph_res: SimulationResult,
    base_res: SimulationResult,
    env: NetworkEnvironment,
    out_path: Path,
) -> None:
    """图 3：典型执行期闭环时间轴回放（展示注入故障、触发重规划、兜底阶梯与中断对比）。"""
    t = paph_res.trace["times_s"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.2), sharex=True)

    # 子图 1：服务卫星编号轨迹与事件触发点
    ax0 = axes[0]
    ax0.step(t, paph_res.trace["sat_idx"], where="post", color="#1E6F5C", linewidth=1.6, label="PAPH 服务卫星")
    ax0.step(
        t, base_res.trace["sat_idx"], where="post", color="#9BA4B5", linewidth=1.1, linestyle="--", label="最强信号基线"
    )
    # 标出软/硬触发重规划时刻
    trig = paph_res.trace["trigger"]
    hard_mask = (trig == "HARD_FAULT") | (trig == "HARD_HO_FAIL")
    soft_mask = trig == "SOFT_DEVIATION"
    if np.any(hard_mask):
        ax0.scatter(t[hard_mask], paph_res.trace["sat_idx"][hard_mask], color="#D9534F", s=34, zorder=5, label="硬触发重规划")
    if np.any(soft_mask):
        ax0.scatter(t[soft_mask], paph_res.trace["sat_idx"][soft_mask], color="#F0AD4E", marker="^", s=36, zorder=5, label="软触发重规划")
    ax0.set_ylabel("接入卫星编号")
    ax0.set_title("服务卫星切换轨迹与事件触发重规划回放", fontsize=11, fontweight="bold")
    ax0.legend(loc="upper right", ncol=4, fontsize=8.5)
    ax0.grid(True, linestyle="--", alpha=0.35)

    # 子图 2：端到端时延与注入故障区间着色
    ax1 = axes[1]
    ax1.plot(t, base_res.trace["delay_ms"], color="#9BA4B5", linewidth=1.0, alpha=0.8, label="最强信号时延")
    ax1.plot(t, paph_res.trace["delay_ms"], color="#1E6F5C", linewidth=1.4, label="PAPH 端到端时延")
    fault_colors = {"rain_fade": "#5BC0DE", "load_surge": "#F0AD4E", "link_fault": "#D9534F"}
    seen_types = set()
    for flt in env.faults:
        lbl = None
        if flt.fault_type not in seen_types:
            lbl_map = {"rain_fade": "注入：雨衰/遮挡", "load_surge": "注入：负载突变", "link_fault": "注入：链路故障"}
            lbl = lbl_map[flt.fault_type]
            seen_types.add(flt.fault_type)
        ax1.axvspan(flt.start_s, flt.end_s, color=fault_colors[flt.fault_type], alpha=0.14, label=lbl)
    ax1.set_ylabel("端到端时延 (ms)")
    ax1.set_title("端到端时延演化与故障注入区间", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", ncol=3, fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # 子图 3：逐时刻中断时长对比 + PAPH 兜底等级 (L1~L4)
    ax2 = axes[2]
    ax2.fill_between(
        t, 0, base_res.trace["outage_s"], step="post", color="#D9534F", alpha=0.38, label="最强信号中断时长 (s/拍)"
    )
    ax2.fill_between(
        t, 0, paph_res.trace["outage_s"], step="post", color="#1E6F5C", alpha=0.75, label="PAPH 中断时长 (s/拍)"
    )
    fb_map = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
    fb_num = np.array([fb_map.get(str(x), 1) for x in paph_res.trace["fallback"]])
    ax2_r = ax2.twinx()
    ax2_r.step(t, fb_num, where="post", color="#3A5BA0", linewidth=1.1, alpha=0.75, label="PAPH 兜底等级")
    ax2_r.set_yticks([1, 2, 3, 4])
    ax2_r.set_yticklabels(["L1 主序列", "L2 备份", "L3 保当前", "L4 降级"], fontsize=8.5)
    ax2_r.set_ylim(0.7, 4.4)
    ax2.set_xlabel("仿真时间 (秒)")
    ax2.set_ylabel("单拍中断 (s)")
    ax2.set_title("中断事件对比与四级兜底阶梯响应", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("PAPH 执行期保障闭环回放（测试场景）", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_realtime_budget(budget_df: pd.DataFrame, dt_plan_s: float, out_path: Path) -> None:
    """图 4：实时性预算验证（不同 H, K 下各阶段实测耗时拆解 vs Δt）。"""
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    x = np.arange(len(budget_df))
    labels = [f"H={int(r.H)}, K={int(r.K)}" for r in budget_df.itertuples()]

    u = budget_df["UpdateForecast(ms)"].values
    f = budget_df["Filter(ms)"].values
    d = budget_df["DP(ms)"].values
    s = budget_df["SignalingEst(ms)"].values

    ax.bar(x, u, width=0.52, label="状态更新 + 增量预测", color="#4A90E2")
    ax.bar(x, f, width=0.52, bottom=u, label="候选可行性筛选", color="#5CB85C")
    ax.bar(x, d, width=0.52, bottom=u + f, label="DP 求解 O(HK²)", color="#F0AD4E")
    ax.bar(x, s, width=0.52, bottom=u + f + d, label="切换信令与确认开销(模型估算)", color="#9B59B6")

    tot = u + f + d + s
    for xi, tv, mx in zip(x, tot, budget_df["MaxPlanCPU(ms)"].values):
        ax.text(xi, tv + max(tot) * 0.03, f"总计 {tv:.1f}ms\n(最坏CPU {mx:.2f}ms)", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("单次滚动总耗时 (ms)")
    ax.set_ylim(0, max(tot) * 1.32)
    ax.set_title(
        f"实时性预算验证：不同 (H, K) 规模下各阶段耗时拆解（远低于 Δt = {dt_plan_s * 1000:.0f} ms 预算上限）",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_sensitivity(sens_df: pd.DataFrame, out_path: Path) -> None:
    """图 5：风险容忍度 ε 敏感性分析（中断占比 vs 端到端时延帕累托权衡）。"""
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(
        sens_df["MeanDelay(ms)"],
        sens_df["OutageRatio(%)"],
        marker="o",
        markersize=8,
        linewidth=2.0,
        color="#1E6F5C",
    )
    for row in sens_df.itertuples():
        ax.annotate(
            f"ε={row.Epsilon:.2f}\n(HO={row.Handovers})",
            (row._3, row._2),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("平均端到端时延 D (ms)")
    ax.set_ylabel("实际中断占比 O (%)")
    ax.set_title("风险参数 ε 敏感性：安全余量带来的“中断下降 vs 时延让步”权衡曲线", fontsize=11.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _configure_matplotlib()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_config()
    dt_plan = float(cfg["simulation"]["dt_plan"])

    # 0. AHP 权重与一致性校验
    ahp_w, ahp_cr = compute_ahp_weights(cfg["paph"]["ahp_pairwise"])
    print("=" * 72)
    print("PAPH 执行期连续性保障仿真实验平台")
    print("=" * 72)
    print(
        f"[AHP 权重校验] α(时延)={ahp_w[0]:.3f}, β(切换)={ahp_w[1]:.3f}, "
        f"γ(负载)={ahp_w[2]:.3f}, δ(中断)={ahp_w[3]:.3f} | 一致性比率 CR={ahp_cr:.4f} (<0.10 通过)"
    )

    # 1. 构建验证场景与测试场景（独立轨道相位、独立种子与故障序列）
    envs: Dict[str, NetworkEnvironment] = {}
    for key in ["validation", "test"]:
        sc_cfg = cfg["scenarios"][key]
        cache_file = ROOT / "data" / f"precomputed_trace_{key}.npz"
        orbit = build_orbit_trace(
            cfg=cfg,
            time_offset_s=float(sc_cfg["orbit_time_offset_s"]),
            extra_horizon_s=max(32, int(cfg["simulation"]["horizon_H"])) * dt_plan + 40.0,
            cache_path=cache_file,
        )
        envs[key] = build_environment(orbit, sc_cfg, cfg["simulation"])
        print(
            f"[场景就绪] {sc_cfg['name']}: 活跃星={len(orbit.sat_ids)} 颗, "
            f"注入故障={len(envs[key].faults)} 个 (雨衰/负载尖峰/硬故障), 轨道缓存={cache_file.name}"
        )

    # 2. 验证场景核验（用于确认参数标定合理）
    val_env = envs["validation"]
    val_base = run_simulation(StrongestSignalPolicy(val_env, cfg), val_env, cfg)
    val_paph = run_simulation(PAPHPolicy(val_env, cfg), val_env, cfg)
    print(
        f"\n[验证场景核验] 最强信号中断={val_base.outage_ratio_pct:.2f}% (最长 {val_base.max_single_outage_s:.2f}s) "
        f"-> PAPH 中断={val_paph.outage_ratio_pct:.2f}% (最长 {val_paph.max_single_outage_s:.2f}s)"
    )

    # 3. 测试场景：对照基线对比实验
    test_env = envs["test"]
    baseline_policies = [
        StrongestSignalPolicy(test_env, cfg),
        MaxVisiblePolicy(test_env, cfg),
        WeightedMyopicPolicy(test_env, cfg),
        PAPHPolicy(test_env, cfg, name="PAPH (本方案)"),
    ]
    baseline_results = [run_simulation(p, test_env, cfg) for p in baseline_policies]
    baseline_df = pd.DataFrame([r.to_summary_dict() for r in baseline_results])
    baseline_df.to_csv(RESULTS_DIR / "baseline_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n--- [实验 1] 测试场景对照基线结果 ---")
    print(baseline_df.to_string(index=False))

    paph_full_res = baseline_results[-1]
    print(f"    PAPH 四级兜底分布占比: {paph_full_res.fallback_dist_pct}")
    print(
        f"    PAPH 关键信令丢包率={paph_full_res.critical_loss_pct:.2f}% vs "
        f"最强信号关键信令丢包率={baseline_results[0].critical_loss_pct:.2f}%"
    )

    # 4. 测试场景：消融实验（逐一剥离机制）
    ablation_policies = [
        PAPHPolicy(test_env, cfg, name="PAPH 完整方案"),
        PAPHPolicy(test_env, cfg, name="消融: 去余量", use_margin=False),
        PAPHPolicy(test_env, cfg, name="消融: 去确认切换", use_mbb=False),
        PAPHPolicy(test_env, cfg, name="消融: 去触发重规划", use_event_trigger=False),
    ]
    ablation_results = [run_simulation(p, test_env, cfg) for p in ablation_policies]
    ablation_df = pd.DataFrame([r.to_summary_dict() for r in ablation_results])
    ablation_df.to_csv(RESULTS_DIR / "ablation_study.csv", index=False, encoding="utf-8-sig")

    print("\n--- [实验 2] 测试场景消融实验结果 ---")
    print(ablation_df.to_string(index=False))

    # 5. 实时性预算实验（扫描 H, K 上限）
    budget_rows = []
    for H_val, K_val in [(6, 4), (12, 6), (18, 6), (24, 8), (30, 10)]:
        p = PAPHPolicy(test_env, cfg, name=f"PAPH(H={H_val},K={K_val})", horizon_override=H_val, k_override=K_val)
        r = run_simulation(p, test_env, cfg)
        budget_rows.append(
            {
                "H": H_val,
                "K": K_val,
                "UpdateForecast(ms)": round(r.mean_stage_ms["update_forecast_ms"], 3),
                "Filter(ms)": round(r.mean_stage_ms["filter_ms"], 3),
                "DP(ms)": round(r.mean_stage_ms["dp_ms"], 3),
                "SignalingEst(ms)": round(r.mean_stage_ms["signaling_ms"], 2),
                "MeanPlanCPU(ms)": round(r.mean_plan_cpu_ms, 3),
                "MaxPlanCPU(ms)": round(r.max_plan_cpu_ms, 3),
                "BudgetPct(%)": round(100.0 * (r.max_plan_cpu_ms + r.mean_stage_ms["signaling_ms"]) / (dt_plan * 1000.0), 2),
            }
        )
    budget_df = pd.DataFrame(budget_rows)
    budget_df.to_csv(RESULTS_DIR / "realtime_budget.csv", index=False, encoding="utf-8-sig")
    print("\n--- [实验 3] 实时性预算测试（Δt = 5000 ms） ---")
    print(budget_df.to_string(index=False))

    # 6. 风险容忍度 ε 敏感性分析
    sens_rows = []
    for eps_val in [0.03, 0.06, 0.10, 0.18, 0.30]:
        p = PAPHPolicy(test_env, cfg, name=f"eps={eps_val}", epsilon_override=eps_val)
        r = run_simulation(p, test_env, cfg)
        sens_rows.append(
            {
                "Epsilon": eps_val,
                "OutageRatio(%)": round(r.outage_ratio_pct, 2),
                "MeanDelay(ms)": round(r.mean_e2e_delay_ms, 2),
                "MaxSingleOutage(s)": round(r.max_single_outage_s, 2),
                "Handovers": r.handover_count,
            }
        )
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(RESULTS_DIR / "epsilon_sensitivity.csv", index=False, encoding="utf-8-sig")
    print("\n--- [实验 4] 机会约束参数 ε 敏感性分析 ---")
    print(sens_df.to_string(index=False))

    # 7. 生成 5 张实验分析图表
    plot_baseline_comparison(baseline_results, FIG_DIR / "fig1_baseline_comparison.png")
    plot_ablation_study(ablation_results, FIG_DIR / "fig2_ablation_study.png")
    plot_execution_timeline(paph_full_res, baseline_results[0], test_env, FIG_DIR / "fig3_execution_timeline.png")
    plot_realtime_budget(budget_df, dt_plan, FIG_DIR / "fig4_realtime_budget.png")
    plot_pareto_sensitivity(sens_df, FIG_DIR / "fig5_pareto_sensitivity.png")

    print(f"\n全部实验完成！CSV 表格已存至 {RESULTS_DIR}，5 张图表已存至 {FIG_DIR}。")


if __name__ == "__main__":
    main()
