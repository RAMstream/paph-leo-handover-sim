"""
轨道几何与离线预计算模块（确定性分量）
======================================
支持两种模式：
1. synthetic（默认）：基于 Walker-Delta 构型的低轨星座物理外推（零外部网络依赖，可完全复现）。
2. tle（可选）：读取 CelesTrak 公开 TLE 星历，利用 sgp4 进行 SGP4 轨道外推。

对应汇报第 6、8、13、14 页：
- 轨道几何属于确定性分量，离线预计算仰角、斜距、多普勒、剩余可见时间与信关站出口可达性，
  在线仅做查表与增量修正（预计算摊销）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

EARTH_RADIUS_KM = 6371.0
MU_EARTH = 398600.4418  # km^3 / s^2
OMEGA_EARTH = 7.2921159e-5  # rad / s
SPEED_OF_LIGHT_KM_S = 299792.458


@dataclass
class OrbitTrace:
    """离线预计算后的完整轨道轨迹数据结构（细粒度微步长 dt_sim）。"""

    times_s: np.ndarray            # (T,) 仿真时刻（秒）
    dt_sim: float                  # 微步长（秒）
    sat_ids: List[str]             # (N,) 窗口内出现过的活跃卫星名称
    elevation_deg: np.ndarray      # (T, N) 对终端仰角（度）
    slant_range_km: np.ndarray     # (T, N) 星地斜距（km）
    doppler_khz: np.ndarray        # (T, N) Ka/Ku 频段归一化多普勒频移估算（kHz）
    rem_vis_sec: np.ndarray        # (T, N) 当前仰角门限下的剩余可见时间（秒）
    gw_reachable: np.ndarray       # (T, N) 地面出口是否可达 (bool)
    isl_hops: np.ndarray           # (T, N) 到达最近可用信关站的星间跳数 (0, 1, 2)
    prop_delay_ms: np.ndarray      # (T, N) 端到端单向传播+星间+地面出口基础时延 (ms)
    downstream_bias_ms: np.ndarray # (N,)   不同轨道面/信关站出口导致的下游基础时延差异 (ms)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            times_s=self.times_s,
            dt_sim=np.array([self.dt_sim]),
            sat_ids=np.array(self.sat_ids),
            elevation_deg=self.elevation_deg,
            slant_range_km=self.slant_range_km,
            doppler_khz=self.doppler_khz,
            rem_vis_sec=self.rem_vis_sec,
            gw_reachable=self.gw_reachable,
            isl_hops=self.isl_hops,
            prop_delay_ms=self.prop_delay_ms,
            downstream_bias_ms=self.downstream_bias_ms,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "OrbitTrace":
        data = np.load(path, allow_pickle=False)
        return cls(
            times_s=data["times_s"],
            dt_sim=float(data["dt_sim"][0]),
            sat_ids=[str(s) for s in data["sat_ids"]],
            elevation_deg=data["elevation_deg"],
            slant_range_km=data["slant_range_km"],
            doppler_khz=data["doppler_khz"],
            rem_vis_sec=data["rem_vis_sec"],
            gw_reachable=data["gw_reachable"],
            isl_hops=data["isl_hops"],
            prop_delay_ms=data["prop_delay_ms"],
            downstream_bias_ms=data["downstream_bias_ms"],
        )


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """将地面点经纬度转换为 ECEF 坐标 (km) 及当地天顶单位向量。"""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    up = np.array([
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ], dtype=np.float64)
    pos = (EARTH_RADIUS_KM + alt_km) * up
    return pos, up


def _compute_elevation_and_range(
    sat_ecef: np.ndarray,
    ground_pos: np.ndarray,
    ground_up: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算所有时刻、所有卫星相对于地面点的仰角(度)与斜距(km)。
    sat_ecef: shape (T, N, 3)
    """
    los = sat_ecef - ground_pos[None, None, :]  # (T, N, 3)
    slant = np.linalg.norm(los, axis=-1)        # (T, N)
    dot_up = np.einsum("tnk,k->tn", los, ground_up)
    sin_el = np.clip(dot_up / np.maximum(slant, 1e-6), -1.0, 1.0)
    elev_deg = np.degrees(np.arcsin(sin_el))
    return elev_deg, slant


def _compute_remaining_visibility(elev_deg: np.ndarray, min_elev_deg: float, dt_sim: float) -> np.ndarray:
    """逆向扫描计算每个时刻每颗卫星在 min_elev_deg 门限下的剩余可见时长（秒）。"""
    T, N = elev_deg.shape
    vis = elev_deg >= min_elev_deg
    rem = np.zeros((T, N), dtype=np.float32)
    running = np.zeros(N, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        running = np.where(vis[t], running + dt_sim, 0.0)
        rem[t] = running
    return rem


def _propagate_walker_delta(
    times_s: np.ndarray,
    time_offset_s: float,
    altitude_km: float,
    inclination_deg: float,
    num_planes: int,
    sats_per_plane: int,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    向量化 Walker-Delta 低轨星座轨道外推（含地球自转）。
    返回: sat_ecef (T, N, 3), sat_ids, plane_indices (N,)
    """
    r = EARTH_RADIUS_KM + altitude_km
    inc = math.radians(inclination_deg)
    n_mean = math.sqrt(MU_EARTH / (r ** 3))  # rad/s

    raan_list = []
    u0_list = []
    sat_ids = []
    plane_idx = []
    phasing = 1
    total_sats = num_planes * sats_per_plane
    for p in range(num_planes):
        raan = 2.0 * math.pi * p / num_planes
        for m in range(sats_per_plane):
            u0 = 2.0 * math.pi * m / sats_per_plane + 2.0 * math.pi * phasing * p / total_sats
            raan_list.append(raan)
            u0_list.append(u0)
            sat_ids.append(f"LEO-P{p:02d}-S{m:02d}")
            plane_idx.append(p)

    raan_arr = np.array(raan_list, dtype=np.float64)[None, :]  # (1, N)
    u0_arr = np.array(u0_list, dtype=np.float64)[None, :]      # (1, N)
    t_abs = (times_s + time_offset_s)[:, None]                 # (T, 1)

    u = u0_arr + n_mean * t_abs
    cos_u, sin_u = np.cos(u), np.sin(u)
    cos_r, sin_r = np.cos(raan_arr), np.sin(raan_arr)
    cos_i, sin_i = math.cos(inc), math.sin(inc)

    x_eci = r * (cos_r * cos_u - sin_r * sin_u * cos_i)
    y_eci = r * (sin_r * cos_u + cos_r * sin_u * cos_i)
    z_eci = r * (sin_u * sin_i)

    # ECI -> ECEF（考虑地球自转）
    gmst = OMEGA_EARTH * t_abs
    cos_g, sin_g = np.cos(gmst), np.sin(gmst)
    x_ecef = x_eci * cos_g + y_eci * sin_g
    y_ecef = -x_eci * sin_g + y_eci * cos_g
    z_ecef = z_eci

    sat_ecef = np.stack([x_ecef, y_ecef, z_ecef], axis=-1)  # (T, N, 3)
    return sat_ecef, sat_ids, np.array(plane_idx, dtype=np.int32)


def _propagate_tle_sgp4(
    tle_path: str | Path,
    times_s: np.ndarray,
    time_offset_s: float,
    max_sats: int = 400,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    可选：使用 sgp4 解析 CelesTrak 公开 TLE 星历，外推 ECEF 坐标。
    需要安装 `pip install sgp4`。
    """
    try:
        from sgp4.api import Satrec, jday
    except ImportError as exc:
        raise RuntimeError(
            "使用 TLE 模式需要安装 sgp4 库：请运行 `pip install sgp4`，或将配置改为 mode: synthetic。"
        ) from exc

    lines = [ln.strip() for ln in Path(tle_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    sats = []
    sat_ids = []
    i = 0
    while i + 2 < len(lines) and len(sats) < max_sats:
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name = lines[i]
            sats.append(Satrec.twoline2rv(lines[i + 1], lines[i + 2]))
            sat_ids.append(name[:18].strip())
            i += 3
        else:
            i += 1

    if not sats:
        raise ValueError(f"未能从 {tle_path} 解析出有效 TLE 数据。")

    # 基准历元：2026-06-01 00:00:00 UTC
    jd0, fr0 = jday(2026, 6, 1, 0, 0, 0.0)
    T = len(times_s)
    N = len(sats)
    sat_ecef = np.zeros((T, N, 3), dtype=np.float64)

    for ti, ts in enumerate(times_s):
        t_sec = ts + time_offset_s
        fr = fr0 + t_sec / 86400.0
        gmst = OMEGA_EARTH * t_sec
        cos_g, sin_g = math.cos(gmst), math.sin(gmst)
        for si, sat in enumerate(sats):
            e, r_teme, _ = sat.sgp4(jd0, fr)
            if e == 0:
                x, y, z = r_teme
                sat_ecef[ti, si, 0] = x * cos_g + y * sin_g
                sat_ecef[ti, si, 1] = -x * sin_g + y * cos_g
                sat_ecef[ti, si, 2] = z
            else:
                sat_ecef[ti, si, :] = [EARTH_RADIUS_KM + 550.0, 0.0, 0.0]

    plane_idx = np.arange(N, dtype=np.int32) % 24
    return sat_ecef, sat_ids, plane_idx


def build_orbit_trace(
    cfg: Dict,
    time_offset_s: float = 0.0,
    extra_horizon_s: float = 120.0,
    cache_path: Optional[str | Path] = None,
) -> OrbitTrace:
    """
    生成或加载离线预计算的轨道轨迹。
    额外多计算 extra_horizon_s，保证仿真末尾的预测窗口 H·Δt 仍具备完整星历。
    """
    if cache_path is not None and Path(cache_path).exists():
        return OrbitTrace.load_npz(cache_path)

    sim_cfg = cfg["simulation"]
    con_cfg = cfg["constellation"]

    dt_sim = float(sim_cfg["dt_sim"])
    duration_s = float(sim_cfg["duration_s"]) + float(extra_horizon_s)
    min_elev_deg = float(sim_cfg["min_elev_deg"])
    times_s = np.arange(0.0, duration_s + 1e-9, dt_sim, dtype=np.float64)

    mode = con_cfg.get("mode", "synthetic")
    if mode == "tle" and Path(con_cfg.get("tle_path", "")).exists():
        sat_ecef, sat_ids, plane_idx = _propagate_tle_sgp4(
            con_cfg["tle_path"], times_s, time_offset_s
        )
    else:
        sat_ecef, sat_ids, plane_idx = _propagate_walker_delta(
            times_s=times_s,
            time_offset_s=time_offset_s,
            altitude_km=float(con_cfg["altitude_km"]),
            inclination_deg=float(con_cfg["inclination_deg"]),
            num_planes=int(con_cfg["num_planes"]),
            sats_per_plane=int(con_cfg["sats_per_plane"]),
        )

    user_pos, user_up = _geodetic_to_ecef(
        float(con_cfg["user_lat_deg"]), float(con_cfg["user_lon_deg"])
    )
    elev_deg, slant_km = _compute_elevation_and_range(sat_ecef, user_pos, user_up)

    # 仅保留仿真窗口内至少出现过一次（仰角 >= min_elev - 8°）的活跃卫星，加速在线查询
    active_mask = np.any(elev_deg >= (min_elev_deg - 8.0), axis=0)
    sat_ecef = sat_ecef[:, active_mask, :]
    elev_deg = elev_deg[:, active_mask]
    slant_km = slant_km[:, active_mask]
    sat_ids = [sid for sid, keep in zip(sat_ids, active_mask) if keep]
    plane_idx = plane_idx[active_mask]

    T, N = elev_deg.shape
    rem_vis_sec = _compute_remaining_visibility(elev_deg, min_elev_deg, dt_sim)

    # 多普勒估算（以 12 GHz Ku 下行为例：f_d = -(v_radial / c) * f_c）
    radial_vel_km_s = np.gradient(slant_km, dt_sim, axis=0)
    doppler_khz = (-radial_vel_km_s / SPEED_OF_LIGHT_KM_S) * 12e6

    # 计算各信关站可达性与星间链路（ISL）跳数及基础端到端传播时延
    gateways = con_cfg.get("gateways", [])
    best_gw_delay_ms = np.full((T, N), np.inf, dtype=np.float64)
    gw_direct_vis = np.zeros((T, N), dtype=bool)
    gw_min_elev = 15.0

    for gw in gateways:
        gw_pos, gw_up = _geodetic_to_ecef(float(gw["lat_deg"]), float(gw["lon_deg"]))
        gw_el, gw_slant = _compute_elevation_and_range(sat_ecef, gw_pos, gw_up)
        can_connect = gw_el >= gw_min_elev
        gw_direct_vis |= can_connect
        # 下行馈电链路传播 + 地面网出口时延
        d_ms = (gw_slant / SPEED_OF_LIGHT_KM_S) * 1000.0 + float(gw.get("base_terrestrial_ms", 10.0))
        best_gw_delay_ms = np.where(can_connect & (d_ms < best_gw_delay_ms), d_ms, best_gw_delay_ms)

    # 未直连信关站的卫星通过 1~2 跳星间链路中继至最近直连卫星
    gw_reachable = gw_direct_vis.copy()
    isl_hops = np.where(gw_direct_vis, 0, 3).astype(np.int32)
    isl_extra_ms = np.zeros((T, N), dtype=np.float64)
    # 不同轨道面赋予轻微下游路由差异（体现“仰角高但下游出口绕路”的选星冲突）
    downstream_bias_ms = (plane_idx % 4).astype(np.float32) * 5.5

    for t in range(T):
        direct_idx = np.where(gw_direct_vis[t])[0]
        if len(direct_idx) == 0:
            continue
        non_direct_idx = np.where(~gw_direct_vis[t])[0]
        if len(non_direct_idx) == 0:
            continue
        # 计算非直连星到所有直连星的欧氏距离 (km)
        pos_nd = sat_ecef[t][non_direct_idx]
        pos_d = sat_ecef[t][direct_idx]
        diff = pos_nd[:, None, :] - pos_d[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)  # (M_nd, M_d)
        min_j = np.argmin(dist_mat, axis=1)
        min_dist = dist_mat[np.arange(len(non_direct_idx)), min_j]
        nearest_gw_sat = direct_idx[min_j]

        for idx_local, sat_i in enumerate(non_direct_idx):
            d_isl = min_dist[idx_local]
            if d_isl <= 1600.0:
                hops = 1
            elif d_isl <= 3200.0:
                hops = 2
            else:
                continue
            gw_reachable[t, sat_i] = True
            isl_hops[t, sat_i] = hops
            # ISL 传输 + 路由转发开销 (每跳约 3.5 ms 处理 + 空间传播)
            isl_extra_ms[t, sat_i] = (d_isl / SPEED_OF_LIGHT_KM_S) * 1000.0 + hops * 3.5
            best_gw_delay_ms[t, sat_i] = best_gw_delay_ms[t, nearest_gw_sat[idx_local]]

    # 端到端往返/单向基础传播时延（采用 RTT 风格 2x 用户上行 + ISL + 馈电 + 地面出口差异）
    user_prop_ms = (slant_km / SPEED_OF_LIGHT_KM_S) * 1000.0
    prop_delay_ms = np.where(
        gw_reachable,
        2.0 * user_prop_ms + isl_extra_ms + best_gw_delay_ms + downstream_bias_ms[None, :],
        999.0,
    )

    trace = OrbitTrace(
        times_s=times_s.astype(np.float32),
        dt_sim=dt_sim,
        sat_ids=sat_ids,
        elevation_deg=elev_deg.astype(np.float32),
        slant_range_km=slant_km.astype(np.float32),
        doppler_khz=doppler_khz.astype(np.float32),
        rem_vis_sec=rem_vis_sec.astype(np.float32),
        gw_reachable=gw_reachable,
        isl_hops=isl_hops,
        prop_delay_ms=prop_delay_ms.astype(np.float32),
        downstream_bias_ms=downstream_bias_ms,
    )

    if cache_path is not None:
        trace.save_npz(cache_path)
    return trace
