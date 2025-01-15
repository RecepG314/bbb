#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QLabel, QLineEdit, QPushButton, QWidget
from PyQt5.QtCore import pyqtSlot

import threading  # ROS2 ve GUI eşzamanlı çalışacak
import numpy as np
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, Empty


class DroneAreaScanner(Node):
    def __init__(self):
        super().__init__('drone_area_scanner')

        self.velocity_publisher = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.photo_trigger_publisher = self.create_publisher(Image, '/simple_drone/bottom/image_raw', 10)
        self.odometry_subscriber = self.create_subscription(Odometry, '/simple_drone/odom', self.odometry_callback, 10)

        self.stop_command_publisher = self.create_publisher(Header, '/stopcommand', 10)  # StopCommand için publisher
        self.finish_command_publisher = self.create_publisher(Header, '/finishcommand', 10)

        self.takeoff_publisher = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.land_publisher = self.create_publisher(Empty, '/simple_drone/land', 10)

        self.declare_parameter('scan_step', 1.0)
        self.declare_parameter('flight_altitude', 2.5)

        self.scan_step = self.get_parameter('scan_step').value
        self.flight_altitude = self.get_parameter('flight_altitude').value

        self.current_position = [0.0, 0.0, 0.0]
        self.position_lock = threading.Lock()

    def odometry_callback(self, msg):
        with self.position_lock:
            self.current_position = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]

    def create_stop_command(self, value):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = str(value)
        return header

    def create_finish_command(self, value):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = str(value)
        return header

    def generate_coverage_path(self, coordinates):
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        waypoints = []
        x = x_min
        direction = 1
        while x <= x_max:
            if direction == 1:
                waypoints.extend([[x, y, self.flight_altitude] for y in np.arange(y_min, y_max + self.scan_step, self.scan_step)])
            else:
                waypoints.extend([[x, y, self.flight_altitude] for y in np.arange(y_max, y_min - self.scan_step, -self.scan_step)])
            x += self.scan_step
            direction *= -1
        return waypoints

    def create_velocity_command(self, x_speed, y_speed, z_speed):
        twist = Twist()
        twist.linear.x = x_speed
        twist.linear.y = y_speed
        twist.linear.z = z_speed
        return twist

    def move_to_waypoint(self, target_position):
        self.get_logger().info(f'Hedefe gidiliyor: {target_position}')
        kp_xy = 0.5
        kp_z = 0.3

        while not self.is_close_enough(self.current_position, target_position):
            with self.position_lock:
                error_x = target_position[0] - self.current_position[0]
                error_y = target_position[1] - self.current_position[1]
                error_z = target_position[2] - self.current_position[2]
                x_speed = np.clip(kp_xy * error_x, -1.0, 1.0)
                y_speed = np.clip(kp_xy * error_y, -1.0, 1.0)
                z_speed = np.clip(kp_z * error_z, -1.0, 1.0)
                velocity_cmd = self.create_velocity_command(x_speed, y_speed, z_speed)
                self.velocity_publisher.publish(velocity_cmd)
            rclpy.spin_once(self, timeout_sec=0.1)

        stop_cmd = self.create_velocity_command(0.0, 0.0, 0.0)
        self.velocity_publisher.publish(stop_cmd)
        self.get_logger().info('Hedefe ulaşıldı!')

        # StopCommand yayınlama
        self.get_logger().info('StopCommand: True')
        self.stop_command_publisher.publish(self.create_stop_command(True))

        # Bekleme süresi (örnek: 1 saniye)
        self.get_clock().sleep_for(rclpy.duration.Duration(seconds=1))

        # StopCommand: False
        self.get_logger().info('StopCommand: False')
        self.stop_command_publisher.publish(self.create_stop_command(False))

    def is_close_enough(self, current, target, threshold=0.2):
        distance = math.sqrt(
            (current[0] - target[0])**2 +
            (current[1] - target[1])**2 +
            (current[2] - target[2])**2
        )
        return distance < threshold

    def start_scanning(self, coordinates):
        self.takeoff_publisher.publish(Empty())
        waypoints = self.generate_coverage_path(coordinates)
        # İlk konumu sakla
        initial_position = self.current_position.copy()

        for waypoint in waypoints:
            self.move_to_waypoint(waypoint)
            self.get_logger().info('Fotoğraf çekildi!')

        initial_position[2] = self.flight_altitude

        # Tarama işlemi tamamlandı, başlangıç konumuna dön
        self.get_logger().info('Tarama tamamlandı, başlangıç konumuna dönülüyor...')
        self.move_to_waypoint(initial_position)

        # Başlangıç konumuna ulaşıldığını kontrol et
        if self.is_close_enough(self.current_position, initial_position):
            self.get_logger().info('Başlangıç konumuna geri dönüldü.')
            self.get_logger().info('FinishCommand: True')
            self.finish_command_publisher.publish(self.create_finish_command(True))
            self.land_publisher.publish(Empty())
        else:
            self.get_logger().error('Başlangıç konumuna ulaşılamadı, FinishCommand yayınlanmadı.')



# GUI için sınıf
class DroneControlGUI(QWidget):
    def __init__(self, drone_scanner):
        super().__init__()
        self.drone_scanner = drone_scanner
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Drone Kontrol Paneli')
        layout = QVBoxLayout()

        self.coord_label = QLabel('Koordinatlar (x1,y1;x2,y2;x3,y3;x4,y4):')
        layout.addWidget(self.coord_label)

        self.coord_input = QLineEdit()
        layout.addWidget(self.coord_input)

        self.start_button = QPushButton('Haritalamayı Başlat')
        self.start_button.clicked.connect(self.on_start_clicked)
        layout.addWidget(self.start_button)

        self.setLayout(layout)

    @pyqtSlot()
    def on_start_clicked(self):
        coord_text = self.coord_input.text()
        try:
            coordinates = [
                [float(x), float(y)] for x, y in
                (coord.split(',') for coord in coord_text.split(';'))
            ]
            threading.Thread(target=self.drone_scanner.start_scanning, args=(coordinates,)).start()
        except ValueError:
            print('Geçersiz koordinat formatı!')

def main(args=None):
    rclpy.init(args=args)
    drone_scanner = DroneAreaScanner()

    app = QApplication(sys.argv)
    gui = DroneControlGUI(drone_scanner)
    gui.show()

    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f'Hata: {e}')
    finally:
        drone_scanner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
