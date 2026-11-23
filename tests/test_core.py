"""
核心模块单元测试（tests/test_core.py）
验证 AHP 一致性、轨道可见性、DP 规划、确认式切换回退与兜底阶梯逻辑。
运行方式：python -m unittest discover -s tests -v
"""

from pathlib import Path
import unittest
import yaml
import numpy as np

from core.planner import MarginDPPlanner, compute_ahp_weights
from core.predictor import RollingPredictor
from env.channel_load import build_environment
from env.orbit import build_orbit_trace
from exec_layer.fallback import FallbackLadder
from exec_layer.handover_fsm import HandoverFSM


ROOT = Path(__file__).resolve().parent.parent


class TestPAPHCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(ROOT / "config" / "scenarios.yaml", "r", encoding="utf-8") as f:
            cls.cfg = yaml.safe_load(f)
        # 短时长用于快速单元测试
        cls.short_cfg = yaml.safe_load(yaml.safe_dump(cls.cfg))
        cls.short_cfg["simulation"]["duration_s"] = 120.0
        cls.orbit = build_orbit_trace(cls.short_cfg, time_offset_s=0.0, extra_horizon_s=100.0)
        cls.env = build_environment(cls.orbit, cls.short_cfg["scenarios"]["validation"], cls.short_cfg["simulation"])

    def test_ahp_weights_consistency(self) -> None:
        w, cr = compute_ahp_weights(self.cfg["paph"]["ahp_pairwise"])
        self.assertAlmostEqual(float(np.sum(w)), 1.0, places=6)
        self.assertLess(cr, 0.10, "AHP 成对比较矩阵一致性比率 CR 必须小于 0.10")

    def test_orbit_visibility_and_delay(self) -> None:
        vis_counts = np.sum(self.orbit.elevation_deg >= self.short_cfg["simulation"]["min_elev_deg"], axis=1)
        self.assertGreater(int(np.min(vis_counts)), 0, "每个时刻至少应有 1 颗可见卫星")
        self.assertGreater(float(np.mean(vis_counts)), 3.0, "中纬度 Walker-Delta 应有多星重叠覆盖")

    def test_planner_produces_primary_and_backup(self) -> None:
        pred = RollingPredictor(self.env, dt_plan=5.0)
        pred.observe_and_calibrate(0)
        fw = pred.forecast(0, horizon_H=12)
        planner = MarginDPPlanner(
            min_elev_deg=25.0,
            max_candidates_K=6,
            ahp_pairwise=self.cfg["paph"]["ahp_pairwise"],
            use_margin=True,
        )
        res = planner.solve(fw, curr_sat=-1)
        self.assertEqual(len(res.primary_seq), 12)
        self.assertIsNotNone(res.backup_sat)
        self.assertNotEqual(res.primary_sat, res.backup_sat)

    def test_mbb_rollback_on_faulty_target(self) -> None:
        fsm = HandoverFSM(dt_sim=0.5, confirm_delay_s=0.5, use_mbb=True)
        # 找两颗在 t=0 可见的星，人为把目标星标为硬故障
        vis = np.where((self.orbit.elevation_deg[0] >= 30.0) & self.orbit.gw_reachable[0])[0]
        self.assertGreaterEqual(len(vis), 2)
        s_curr, s_bad = int(vis[0]), int(vis[1])
        orig = bool(self.env.hard_fault_mask[1, s_bad])
        self.env.hard_fault_mask[1, s_bad] = True
        try:
            o1 = fsm.request_handover(s_curr, s_bad, tick=0, env=self.env)
            self.assertEqual(o1.active_sat, s_curr)
            self.assertEqual(o1.ho_interruption_s, 0.0)
            o2 = fsm.tick_pending(s_curr, tick=1, env=self.env)
            self.assertTrue(o2.confirm_failed)
            self.assertTrue(o2.rolled_back)
            self.assertEqual(o2.active_sat, s_curr)
        finally:
            self.env.hard_fault_mask[1, s_bad] = orig

    def test_fallback_ladder_l2_on_primary_fail(self) -> None:
        fb = FallbackLadder(enable_fallback=True)
        vis = np.where(
            (self.orbit.elevation_deg[0] >= 30.0)
            & self.orbit.gw_reachable[0]
            & (~self.env.hard_fault_mask[0])
            & (self.env.true_load[0] < 0.80)
        )[0]
        if len(vis) >= 3:
            dec = fb.resolve(
                tick=0,
                curr_sat=int(vis[0]),
                primary_sat=int(vis[1]),
                backup_sat=int(vis[2]),
                env=self.env,
                primary_failed=True,
            )
            self.assertEqual(dec.level, "L2")
            self.assertEqual(dec.target_sat, int(vis[2]))


if __name__ == "__main__":
    unittest.main()
