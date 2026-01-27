import math
import time
from typing import Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from boxbunny_msgs.msg import ImuDebug

try:
    from smbus2 import SMBus
except Exception:  # pragma: no cover
    SMBus = None


MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B


class Mpu6050Node(Node):
    def __init__(self) -> None:
        super().__init__("mpu6050_node")

        self.declare_parameter("i2c_bus", 1)
        self.declare_parameter("i2c_address", MPU6050_ADDR)
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("accel_bias", [0.0, 0.0, 0.0])
        self.declare_parameter("gyro_bias", [0.0, 0.0, 0.0])

        self.imu_pub = self.create_publisher(Imu, "imu/data", 10)
        self.debug_pub = self.create_publisher(ImuDebug, "imu/debug", 10)

        self._bus = None
        self._init_device()

        rate_hz = float(self.get_parameter("rate_hz").value)
        self.timer = self.create_timer(1.0 / rate_hz, self._tick)

        self.get_logger().info("MPU6050 node initialized")

    def _init_device(self) -> None:
        if SMBus is None:
            self.get_logger().error("smbus2 not available. Install via requirements.txt")
            return

        bus_num = int(self.get_parameter("i2c_bus").value)
        address = int(self.get_parameter("i2c_address").value)
        self._bus = SMBus(bus_num)

        # Wake up device
        self._bus.write_byte_data(address, PWR_MGMT_1, 0x00)
        self._bus.write_byte_data(address, SMPLRT_DIV, 0x07)
        self._bus.write_byte_data(address, CONFIG, 0x03)
        self._bus.write_byte_data(address, GYRO_CONFIG, 0x00)   # ±250 dps
        self._bus.write_byte_data(address, ACCEL_CONFIG, 0x00)  # ±2g

    def _read_word(self, address: int, reg: int) -> int:
        high = self._bus.read_byte_data(address, reg)
        low = self._bus.read_byte_data(address, reg + 1)
        value = (high << 8) + low
        if value >= 0x8000:
            value = -((65535 - value) + 1)
        return value

    def _read_accel_gyro(self) -> Tuple[float, float, float, float, float, float]:
        address = int(self.get_parameter("i2c_address").value)
        ax = self._read_word(address, ACCEL_XOUT_H)
        ay = self._read_word(address, ACCEL_XOUT_H + 2)
        az = self._read_word(address, ACCEL_XOUT_H + 4)
        gx = self._read_word(address, ACCEL_XOUT_H + 8)
        gy = self._read_word(address, ACCEL_XOUT_H + 10)
        gz = self._read_word(address, ACCEL_XOUT_H + 12)

        # Convert to m/s^2 and rad/s
        accel_scale = 16384.0
        gyro_scale = 131.0
        ax_m = (ax / accel_scale) * 9.80665
        ay_m = (ay / accel_scale) * 9.80665
        az_m = (az / accel_scale) * 9.80665
        gx_r = math.radians(gx / gyro_scale)
        gy_r = math.radians(gy / gyro_scale)
        gz_r = math.radians(gz / gyro_scale)
        return ax_m, ay_m, az_m, gx_r, gy_r, gz_r

    def _tick(self) -> None:
        if self._bus is None:
            return

        ax, ay, az, gx, gy, gz = self._read_accel_gyro()
        accel_bias = self.get_parameter("accel_bias").value
        gyro_bias = self.get_parameter("gyro_bias").value

        ax -= float(accel_bias[0])
        ay -= float(accel_bias[1])
        az -= float(accel_bias[2])
        gx -= float(gyro_bias[0])
        gy -= float(gyro_bias[1])
        gz -= float(gyro_bias[2])

        imu = Imu()
        imu.header.stamp = self.get_clock().now().to_msg()
        imu.linear_acceleration.x = float(ax)
        imu.linear_acceleration.y = float(ay)
        imu.linear_acceleration.z = float(az)
        imu.angular_velocity.x = float(gx)
        imu.angular_velocity.y = float(gy)
        imu.angular_velocity.z = float(gz)
        self.imu_pub.publish(imu)

        debug = ImuDebug()
        debug.stamp = imu.header.stamp
        debug.ax = float(ax)
        debug.ay = float(ay)
        debug.az = float(az)
        debug.gx = float(gx)
        debug.gy = float(gy)
        debug.gz = float(gz)
        self.debug_pub.publish(debug)


def main() -> None:
    rclpy.init()
    node = Mpu6050Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
