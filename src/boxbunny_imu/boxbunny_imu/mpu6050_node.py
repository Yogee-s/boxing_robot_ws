#!/usr/bin/env python3
"""
MPU6050 IMU Sensor Node.

This module provides a ROS 2 interface for the MPU6050 6-axis IMU sensor
(accelerometer + gyroscope) commonly used in boxing gloves for motion
tracking and punch detection.

The node reads raw sensor data via I2C, applies calibration biases,
and publishes standard ROS Imu messages along with debug information.

Hardware Requirements:
    - MPU6050 sensor connected via I2C
    - smbus2 Python library for I2C communication

ROS 2 Topics:
    Publishers:
        - imu/data (sensor_msgs/Imu): Standard IMU data with acceleration and angular velocity
        - imu/debug (ImuDebug): Debug message with raw axis values

Parameters:
    - i2c_bus (int): I2C bus number (default: 1)
    - i2c_address (int): I2C device address (default: 0x68)
    - rate_hz (float): Sensor polling rate in Hz (default: 50.0)
    - accel_bias (list): Accelerometer calibration offsets [x, y, z]
    - gyro_bias (list): Gyroscope calibration offsets [x, y, z]
"""

import math
import time
from typing import Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from boxbunny_msgs.msg import ImuDebug

try:
    from smbus2 import SMBus
except ImportError:  # pragma: no cover
    SMBus = None


# MPU6050 Register Addresses
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B


class Mpu6050Node(Node):
    """
    ROS 2 node for MPU6050 IMU sensor data acquisition.
    
    Reads accelerometer and gyroscope data from the MPU6050 sensor
    via I2C and publishes it as standard ROS Imu messages.
    
    Attributes:
        imu_pub: Publisher for standard Imu messages.
        debug_pub: Publisher for debug information.
        _bus: SMBus instance for I2C communication.
        timer: Timer for periodic sensor polling.
    """
    
    def __init__(self) -> None:
        """Initialize the MPU6050 sensor node."""
        super().__init__("mpu6050_node")

        # Declare ROS parameters
        self.declare_parameter("i2c_bus", 7)
        self.declare_parameter("i2c_address", MPU6050_ADDR)
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("accel_bias", [0.0, 0.0, 0.0])
        self.declare_parameter("gyro_bias", [0.0, 0.0, 0.0])

        # Set up publishers
        self.imu_pub = self.create_publisher(Imu, "imu/data", 10)
        self.debug_pub = self.create_publisher(ImuDebug, "imu/debug", 10)

        # Initialize I2C connection
        self._bus = None
        self._init_device()

        # Create polling timer
        rate_hz = float(self.get_parameter("rate_hz").value)
        self.timer = self.create_timer(1.0 / rate_hz, self._tick)

        self.get_logger().info(
            f"MPU6050 node initialized - polling at {rate_hz} Hz"
        )

    def _init_device(self) -> None:
        """
        Initialize the MPU6050 sensor hardware.

        Configures the I2C bus connection and sets up the sensor
        with appropriate sampling rate and sensitivity settings.
        If the device is not reachable, logs a warning and leaves
        self._bus as None so _tick becomes a no-op until retry.
        """
        if SMBus is None:
            self.get_logger().error(
                "smbus2 library not available. "
                "Install via: pip install smbus2"
            )
            return

        bus_num = int(self.get_parameter("i2c_bus").value)
        address = int(self.get_parameter("i2c_address").value)

        try:
            bus = SMBus(bus_num)

            # Wake up device (clear sleep bit)
            bus.write_byte_data(address, PWR_MGMT_1, 0x00)

            # Set sample rate divider (1kHz / (1 + 7) = 125Hz internal rate)
            bus.write_byte_data(address, SMPLRT_DIV, 0x07)

            # Set DLPF (Digital Low Pass Filter) for noise reduction
            bus.write_byte_data(address, CONFIG, 0x03)

            # Configure gyroscope: ±250 degrees/second (highest sensitivity)
            bus.write_byte_data(address, GYRO_CONFIG, 0x00)

            # Configure accelerometer: ±2g (highest sensitivity)
            bus.write_byte_data(address, ACCEL_CONFIG, 0x00)

            self._bus = bus
        except OSError as e:
            self.get_logger().warn(
                f"MPU6050 not reachable on bus {bus_num} addr 0x{address:02X}: {e}. "
                "Will retry every 5 seconds..."
            )
            self._bus = None
            if not hasattr(self, '_retry_timer'):
                self._retry_timer = self.create_timer(5.0, self._retry_init)

    def _retry_init(self) -> None:
        """Retry connecting to the MPU6050 sensor."""
        self._init_device()
        if self._bus is not None:
            self.get_logger().info("MPU6050 connected successfully.")
            self._retry_timer.cancel()

    def _read_word(self, address: int, reg: int) -> int:
        """
        Read a 16-bit signed value from two consecutive registers.
        
        Args:
            address: I2C device address.
            reg: Starting register address.
            
        Returns:
            Signed 16-bit integer value.
        """
        high = self._bus.read_byte_data(address, reg)
        low = self._bus.read_byte_data(address, reg + 1)
        value = (high << 8) + low
        
        # Convert to signed value (two's complement)
        if value >= 0x8000:
            value = -((65535 - value) + 1)
        return value

    def _read_accel_gyro(self) -> Tuple[float, float, float, float, float, float]:
        """
        Read all accelerometer and gyroscope axes from the sensor.
        
        Returns:
            Tuple of (ax, ay, az, gx, gy, gz) with accelerometer values
            in m/s² and gyroscope values in rad/s.
        """
        address = int(self.get_parameter("i2c_address").value)
        
        # Read raw accelerometer values
        ax = self._read_word(address, ACCEL_XOUT_H)
        ay = self._read_word(address, ACCEL_XOUT_H + 2)
        az = self._read_word(address, ACCEL_XOUT_H + 4)
        
        # Read raw gyroscope values (offset by 8 bytes from accelerometer)
        gx = self._read_word(address, ACCEL_XOUT_H + 8)
        gy = self._read_word(address, ACCEL_XOUT_H + 10)
        gz = self._read_word(address, ACCEL_XOUT_H + 12)

        # Convert to SI units
        # Accelerometer: 16384 LSB/g at ±2g setting, convert to m/s²
        accel_scale = 16384.0
        ax_m = (ax / accel_scale) * 9.80665
        ay_m = (ay / accel_scale) * 9.80665
        az_m = (az / accel_scale) * 9.80665
        
        # Gyroscope: 131 LSB/(deg/s) at ±250°/s setting, convert to rad/s
        gyro_scale = 131.0
        gx_r = math.radians(gx / gyro_scale)
        gy_r = math.radians(gy / gyro_scale)
        gz_r = math.radians(gz / gyro_scale)
        
        return ax_m, ay_m, az_m, gx_r, gy_r, gz_r

    def _tick(self) -> None:
        """
        Timer callback to read sensor data and publish messages.
        
        Reads accelerometer and gyroscope data, applies calibration
        biases, and publishes both Imu and debug messages.
        """
        if self._bus is None:
            return

        # Read raw sensor data
        ax, ay, az, gx, gy, gz = self._read_accel_gyro()
        
        # Apply calibration biases
        accel_bias = self.get_parameter("accel_bias").value
        gyro_bias = self.get_parameter("gyro_bias").value

        ax -= float(accel_bias[0])
        ay -= float(accel_bias[1])
        az -= float(accel_bias[2])
        gx -= float(gyro_bias[0])
        gy -= float(gyro_bias[1])
        gz -= float(gyro_bias[2])

        # Build and publish standard Imu message
        imu = Imu()
        imu.header.stamp = self.get_clock().now().to_msg()
        imu.linear_acceleration.x = float(ax)
        imu.linear_acceleration.y = float(ay)
        imu.linear_acceleration.z = float(az)
        imu.angular_velocity.x = float(gx)
        imu.angular_velocity.y = float(gy)
        imu.angular_velocity.z = float(gz)
        self.imu_pub.publish(imu)

        # Build and publish debug message
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
    """Entry point for the MPU6050 IMU node."""
    rclpy.init()
    node = Mpu6050Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass  # Already shut down


if __name__ == "__main__":
    main()
