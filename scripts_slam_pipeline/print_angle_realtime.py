"""
实时打印 AS5600 角度读数。

用法:
    python scripts_slam_pipeline/print_angle_realtime.py
    python scripts_slam_pipeline/print_angle_realtime.py --hz 30 --avg 5

说明:
    - raw: AS5600 原始读数，范围约为 [0, 4095]
    - deg: 换算后的角度，范围约为 [0, 360)
    - avg_deg: 最近 N 次读数的滑动平均角度（N 由 --avg 指定）
"""

import argparse
import signal
import sys
import time
from collections import deque

import smbus


DEVICE_AS5600 = 0x36
REGISTER_RAW_ANGLE = 0x0C


def read_raw_angle(bus: smbus.SMBus) -> int:
    """读取 AS5600 原始角度值（12-bit）。"""
    data = bus.read_i2c_block_data(DEVICE_AS5600, REGISTER_RAW_ANGLE, 2)
    return ((data[0] << 8) | data[1]) & 0x0FFF


def raw_to_deg(raw: int) -> float:
    """将 0-4095 原始值映射到 0-360 度。"""
    return (raw * 360.0) / 4096.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="实时打印 AS5600 angle 到控制台")
    parser.add_argument("--hz", type=float, default=20.0, help="打印频率，默认 20 Hz")
    parser.add_argument(
        "--avg",
        type=int,
        default=1,
        help="滑动平均窗口大小，默认 1（不平滑）",
    )
    parser.add_argument("--bus", type=int, default=1, help="I2C bus 编号，默认 1")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.hz <= 0:
        print("[ERROR] --hz 必须大于 0")
        return 2
    if args.avg <= 0:
        print("[ERROR] --avg 必须大于 0")
        return 2

    period = 1.0 / args.hz
    smooth_window = deque(maxlen=args.avg)
    stop = {"flag": False}

    def _handle_stop(_signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        bus = smbus.SMBus(args.bus)
    except Exception as exc:
        print(f"[ERROR] 打开 I2C bus {args.bus} 失败: {exc}")
        return 1

    print(
        f"[INFO] 开始实时读取 AS5600: bus={args.bus}, hz={args.hz}, avg={args.avg}"
    )
    print("[INFO] 按 Ctrl+C 退出")

    try:
        while not stop["flag"]:
            t0 = time.monotonic()
            ts = time.time()

            try:
                raw = read_raw_angle(bus)
            except Exception as exc:
                print(f"[WARN] 读取失败 @ {ts:.6f}: {exc}")
                time.sleep(period)
                continue

            deg = raw_to_deg(raw)
            smooth_window.append(deg)
            avg_deg = sum(smooth_window) / len(smooth_window)

            print(
                f"ts={ts:.6f} | raw={raw:4d} | deg={deg:7.3f} | avg_deg={avg_deg:7.3f}",
                flush=True,
            )

            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        try:
            bus.close()
        except Exception:
            pass
        print("[INFO] 已退出 angle 实时打印")

    return 0


if __name__ == "__main__":
    sys.exit(main())
