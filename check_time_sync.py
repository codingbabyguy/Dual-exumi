#!/usr/bin/env python3
from datetime import datetime, timezone, timedelta

# VR pose 时间戳
vr_start = 1774440513.9008577

# GoPro 视频时间（从 exiftool)
# Create Date: 2026:03:25 20:09:31  (这是本地时间，默认 UTC+8)
gopro_local_str = "2026:03:25 20:09:31"
gopro_dt = datetime.strptime(gopro_local_str, "%Y:%m:%d %H:%M:%S")

# 假设 GoPro 使用 UTC+8 时区
gopro_utc = gopro_dt - timedelta(hours=8)
gopro_timestamp = gopro_utc.replace(tzinfo=timezone.utc).timestamp()

# VR 时间的日期
vr_dt = datetime.fromtimestamp(vr_start, tz=timezone.utc)

print("=== 时间对比 ===")
print(f"VR pose 时间戳: {vr_start}")
print(f"VR 对应日期: {vr_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print()
print(f"GoPro 本地时间: {gopro_local_str} (假设 UTC+8)")
print(f"GoPro 转为 UTC: {gopro_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"GoPro timestamp: {gopro_timestamp:.1f}")
print()
print(f"时间差: {gopro_timestamp - vr_start:.1f} 秒")
print()
print("**问题分析**:")
if abs(gopro_timestamp - vr_start) < 5:
    print("✓ 时间戳匹配！")
else:
    print(f"❌ 时间差超过 5 秒，说明设备时钟不同步")
    if gopro_timestamp > vr_start:
        print(f"   GoPro 比 VR (Orange Pi) 快 {gopro_timestamp - vr_start:.1f} 秒")
    else:
        print(f"   VR (Orange Pi) 比 GoPro 快 {vr_start - gopro_timestamp:.1f} 秒")
