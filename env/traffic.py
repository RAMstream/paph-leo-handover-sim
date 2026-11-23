"""
双优先级业务流模型（服务于兜底阶梯 L4：业务降级）
================================================
对应汇报第 12 页：
- 首版区分两类业务：信令/关键数据（高优先级）与普通业务（尽力而为，可排队兜底）。
- 正常模式（L1/L2/L3）：两类业务共同发送。
- 业务降级模式（L4）：优先保障信令与关键数据不中断，普通业务暂存队列待链路恢复后排空。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrafficStats:
    """业务层连续性统计结果。"""

    critical_sent: int = 0
    critical_dropped: int = 0
    normal_sent: int = 0
    normal_queued_ticks: int = 0
    normal_dropped: int = 0
    max_normal_queue: int = 0

    @property
    def critical_loss_rate(self) -> float:
        total = self.critical_sent + self.critical_dropped
        return self.critical_dropped / max(total, 1)

    @property
    def normal_loss_rate(self) -> float:
        total = self.normal_sent + self.normal_dropped
        return self.normal_dropped / max(total, 1)


class DualPriorityTrafficModel:
    """模拟每个微步长产生的关键信令包与普通数据包。"""

    def __init__(self, max_queue_packets: int = 80) -> None:
        self.max_queue_packets = max_queue_packets
        self.normal_queue: int = 0
        self.stats = TrafficStats()

    def step(
        self,
        link_up: bool,
        degraded_l4: bool,
        sat_load: float,
        snr_db: float,
        snr_th_db: float,
    ) -> None:
        """
        每个仿真微步长产生 1 个关键信令包和 3 个普通数据包。
        - 若链路完全断开（且非可保底降质）：关键包丢弃，普通包进入有限缓冲区（满则丢弃）。
        - 若处于 L4 业务降级模式：即使负载偏高（0.84~0.95）或处于边缘弱信号（低阶编码 MCS），
          仍优先挤出带宽保障关键信令送达，普通业务排队缓冲。
        """
        crit_gen = 1
        norm_gen = 3

        if not link_up:
            # 即使常规高速率解调失败，若处于 L4 降级且 SNR 仍在低阶保底编码门限内（th - 1.5 dB）
            # 且未发生硬件断连，可抢救性送出窄带信令
            if degraded_l4 and snr_db >= (snr_th_db - 1.5) and sat_load < 0.97:
                self.stats.critical_sent += crit_gen
            else:
                self.stats.critical_dropped += crit_gen
            self._enqueue_normal(norm_gen)
            return

        if degraded_l4:
            # L4 业务降级：关键业务 100% 优先通过，普通业务排队兜底
            self.stats.critical_sent += crit_gen
            self._enqueue_normal(norm_gen)
            # 如有少量余量，缓慢释放 1 个普通包
            if sat_load < 0.90 and self.normal_queue > 0:
                self.normal_queue -= 1
                self.stats.normal_sent += 1
        else:
            # 正常链路：关键业务发送，普通业务正常发送并顺带排空积压队列
            self.stats.critical_sent += crit_gen
            self.stats.normal_sent += norm_gen
            drain = min(self.normal_queue, 4)
            self.normal_queue -= drain
            self.stats.normal_sent += drain

        self.stats.max_normal_queue = max(self.stats.max_normal_queue, self.normal_queue)
        if self.normal_queue > 0:
            self.stats.normal_queued_ticks += 1

    def _enqueue_normal(self, count: int) -> None:
        space = self.max_queue_packets - self.normal_queue
        accepted = min(count, max(space, 0))
        dropped = count - accepted
        self.normal_queue += accepted
        self.stats.normal_dropped += dropped
        self.stats.max_normal_queue = max(self.stats.max_normal_queue, self.normal_queue)
        if self.normal_queue > 0:
            self.stats.normal_queued_ticks += 1
