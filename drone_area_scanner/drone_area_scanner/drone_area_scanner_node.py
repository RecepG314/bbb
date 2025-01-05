#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
import threading  # threading import'ı buraya eklendi
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

class DroneAreaScanner(Node):
    def __init__(self):
        super().__init__('drone_area_scanner')

        # Hız ve fotoğraf publisher'ları
        self.velocity_publisher = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.photo_trigger_publisher = self.create_publisher(Image, '/simple_drone/bottom/image_raw', 10)

        # Odometri subscriber'ı
        self.odometry_subscriber = self.create_subscription(
            Odometry, 
            '/simple_drone/odom', 
            self.odometry_callback, 
            10
        )

        # Parametre tanımları
        self.declare_parameter('scan_step', 1.0)  # Her adımda 5 metre
        self.declare_parameter('flight_altitude', 3.0)  # 10 metre yükseklik

        self.scan_step = self.get_parameter('scan_step').value
        self.flight_altitude = self.get_parameter('flight_altitude').value

        # Drone'un anlık konumu
        self.current_position = [0.0, 0.0, 0.0]
        self.position_lock = threading.Lock()

    def odometry_callback(self, msg):
        """Odometri verilerini güncelle"""
        with self.position_lock:
            self.current_position = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]

    def generate_coverage_path(self, coordinates):
        """
        Verilen koordinatları kullanarak alan tarama rotası oluştur

        :param coordinates: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] formatında koordinatlar
        :return: Tarama için waypoint listesi
        """
        # Koordinatları sınırlayan dikdörtgeni hesapla
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Serpantin tarama algoritması
        waypoints = []
        x = x_min
        direction = 1  # 1: yukarı, -1: aşağı

        while x <= x_max:
            if direction == 1:
                waypoints.extend([[x, y, self.flight_altitude] for y in np.arange(y_min, y_max + self.scan_step, self.scan_step)])
            else:
                waypoints.extend([[x, y, self.flight_altitude] for y in np.arange(y_max, y_min - self.scan_step, -self.scan_step)])

            x += self.scan_step
            direction *= -1

        return waypoints

    def create_velocity_command(self, x_speed, y_speed, z_speed):
        """Hız komutu oluştur"""
        twist = Twist()
        twist.linear.x = x_speed
        twist.linear.y = y_speed
        twist.linear.z = z_speed
        return twist

    def take_photo(self):
        """Fotoğraf çekme ve AI modeline gönderme"""
        # Fotoğraf çekme simülasyonu (örneğin boş bir numpy array)
        photo_data = np.zeros((480, 640, 3), dtype=np.uint8)  # Örnek: 480x640 siyah bir görüntü

        # OpenCV kullanarak bir görüntüyü yapay zeka modeline gönder
        photo_msg = Image()
        photo_msg.header = Header()
        photo_msg.header.stamp = self.get_clock().now().to_msg()
        photo_msg.height = photo_data.shape[0]
        photo_msg.width = photo_data.shape[1]
        photo_msg.encoding = 'rgb8'
        photo_msg.step = photo_data.shape[1] * 3
        photo_msg.data = photo_data.tobytes()

        self.photo_trigger_publisher.publish(photo_msg)
        self.get_logger().info('Fotoğraf çekildi ve AI modeline gönderildi.')                               

    def move_to_waypoint(self, target_position):
        """Hedef noktaya doğru hassas hareket et"""
        self.get_logger().info(f'Hedefe gidiliyor: {target_position}')

        # PID kontrolcü parametreleri
        kp_xy = 0.5  # Oransal kazanç
        kp_z = 0.3

        while not self.is_close_enough(self.current_position, target_position):
            with self.position_lock:
                # X ve Y eksenlerinde hata hesaplama
                error_x = target_position[0] - self.current_position[0]
                error_y = target_position[1] - self.current_position[1]
                error_z = target_position[2] - self.current_position[2]

                # PID kontrolcü ile hız hesaplama
                x_speed = kp_xy * error_x
                y_speed = kp_xy * error_y
                z_speed = kp_z * error_z

                # Hız sınırlandırması
                x_speed = np.clip(x_speed, -1.0, 1.0)
                y_speed = np.clip(y_speed, -1.0, 1.0)
                z_speed = np.clip(z_speed, -1.0, 1.0)

                # Hız komutunu gönder
                velocity_cmd = self.create_velocity_command(x_speed, y_speed, z_speed)
                self.velocity_publisher.publish(velocity_cmd)

            rclpy.spin_once(self, timeout_sec=0.1)

        # Dur komutu gönder
        stop_cmd = self.create_velocity_command(0.0, 0.0, 0.0)
        self.velocity_publisher.publish(stop_cmd)
        self.get_logger().info('Hedefe ulaşıldı!')

    def is_close_enough(self, current, target, threshold=0.2):
        """Hedefe yakınlığı hassas bir şekilde kontrol et"""
        distance = math.sqrt(
            (current[0] - target[0])**2 +
            (current[1] - target[1])**2 +
            (current[2] - target[2])**2
        )
        return distance < threshold

    def start_scanning(self, coordinates):
        """Alan taramasını başlat"""
        waypoints = self.generate_coverage_path(coordinates)

        for waypoint in waypoints:
            self.move_to_waypoint(waypoint)
            self.take_photo()


def main(args=None):
    rclpy.init(args=args)

    drone_scanner = DroneAreaScanner()

    # Örnek koordinatlar (gerçek koordinatlarınızla değiştirebilirsiniz)
    example_coords = [
        [5, -2],   # Sol alt köşe
        [5, -7],   # Sol üst köşe
        [-1, -7],   # Sağ üst köşe
        [-1, -2]    # Sağ alt köşe
    ]

    try:
        drone_scanner.start_scanning(example_coords)
    except Exception as e:
        drone_scanner.get_logger().error(f'Hata oluştu: {str(e)}')
    finally:
        drone_scanner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()