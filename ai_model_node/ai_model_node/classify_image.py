#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import joblib
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool, Header 
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from rclpy.timer import Timer
from datetime import datetime
import json
import cv2
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import os

from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class AIModelNode(Node):
    def __init__(self):
        super().__init__('ai_model_node')

        self.get_logger().info("AI Model Node başlatıldı.")

        # ResNet modelini yükle
        self.device = torch.device('cpu')
        self.resnet_model = models.resnet50(pretrained=True).to(self.device)
        self.resnet_model.eval()
        self.get_logger().info("ResNet modeli yüklendi.")

        # SVM modelini yükle
        self.svm_model = joblib.load("/home/qwer/ros2_ws/src/sjtu_drone/ai_model_node/data/svm_model_resnet50.joblib")
        self.get_logger().info("SVM modeli yüklendi.")

        # U-NET modelini yükle
        self.unet_model = load_model("/home/qwer/ros2_ws/src/sjtu_drone/ai_model_node/data/wheat_segmentation_model.h5")
        self.get_logger().info("U-NET modeli yüklendi.")

        # Görüntü işleme dönüşümleri
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # CvBridge ile ROS görüntülerini işleyin
        self.bridge = CvBridge()

        self.prediction_map = {
            0: "Sağlıklı (Healthy)",
            1: "Sarı Pas (Leaf Rust)",
            2: "Rastık (Smut)",
            3: "Kara Pas (Stem Rust)"
        }

        # State variables
        self.latest_image = None
        self.current_position = None
        self.stop_command = False
        self.results_list = []

        # Publishers
        self.result_publisher = self.create_publisher(String, '/ai_model/result', 10)
        self.image_publisher = self.create_publisher(Image, '/processed_image', 10)

        # Subscribers
        self.image_subscriber = self.create_subscription(
            Image,
            '/simple_drone/bottom/image_raw',
            self.image_callback,
            10
        )
        
        self.position_subscriber = self.create_subscription(
            Odometry,
            '/simple_drone/odom',
            self.position_callback,
            10
        )

        self.stop_command_subscriber = self.create_subscription(
            Header,
            '/stopcommand',
            self.stop_command_callback,
            10
        )

    def position_callback(self, msg):
        """Konum bilgisini güncelle"""
        self.current_position = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z
        }

    def stop_command_callback(self, msg):
        """Stop command durumunu güncelle ve işlemi tetikle."""
        try:
            self.stop_command = msg.frame_id.lower() == 'true'  # Header'dan True/False çıkar
            
            if self.stop_command and self.latest_image is not None:
                self.process_image()
        except Exception as e:
            self.get_logger().error(f"StopCommand işleme hatası: {e}")
    
    def process_image(self):
        """Görüntüyü işle ve sonuçları kaydet."""
        try:
            self.get_logger().info(f"process image")
            image_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")

            # U-NET ile görüntüyü maskele
            masked_image = self.apply_unet(cv_image)

            # ResNet ile öznitelik çıkar ve tahmin yap
            feature_vector = self.extract_features(masked_image)
            prediction = self.svm_model.predict(feature_vector.reshape(1, -1))
            predicted_label = self.prediction_map[prediction[0]]

            result_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Sonuçları kaydet
            result_data = {
                'prediction': predicted_label,
                'position': self.current_position,
                'image_timestamp': image_timestamp,
                'result_timestamp': result_timestamp,
                'image': self.latest_image,  # Orijinal görüntü
                'masked_image': masked_image  # İşlenmiş görüntü
            }
            self.results_list.append(result_data)

            # Sonucu yayınla
            result_msg = String()
            result_msg.data = f"Tahmin edilen sınıf: {predicted_label}"
            self.result_publisher.publish(result_msg)

            # İşlenmiş görüntüyü yayınla
            self.publish_image(masked_image)

        except Exception as e:
            self.get_logger().error(f"Görüntü işleme hatası: {e}")
        finally:
            self.latest_image = None

    def apply_unet(self, image):
        """U-NET modeli ile görüntüyü maskele."""
        try:
            img_size = (128, 128)
            resized_image = cv2.resize(image, img_size) / 255.0  # Normalize
            input_image = np.expand_dims(resized_image, axis=0)  # Batch boyutu ekle

            # U-NET ile tahmin yap
            prediction = self.unet_model.predict(input_image)[0]
            prediction_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]))

            # Binary maske oluştur
            threshold = 0.18
            binary_mask = (prediction_resized > threshold).astype(np.uint8)

            # Buğdayları maskele: Sadece buğdaylar görünsün, geri kalan her yer siyah olsun
            masked_image = image.copy()
            masked_image[binary_mask == 1] = [0,0,0]

            return masked_image
        except Exception as e:
            self.get_logger().error(f"U-NET işleme hatası: {e}")
            return image

    def image_callback(self, msg):
        """Görüntüyü al ve kaydet."""
        self.latest_image = msg

    # Ana sınıfın save_results_to_file metodunu güncelle
    def save_results_to_file(self):
        """Sonuçları txt dosyasına kaydet ve UI'ı başlat"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_results_{timestamp}.txt"
            
            # Her bir sonuç için görüntüyü kaydet
            for i, result in enumerate(self.results_list):
                # Orijinal görüntüyü kaydet
                cv_image = self.bridge.imgmsg_to_cv2(result['image'], "bgr8")
                image_filename = f"original_image_{timestamp}_{i}.jpg"
                cv2.imwrite(image_filename, cv_image)
                result['image_path'] = image_filename

                # İşlenmiş görüntüyü kaydet
                masked_image_filename = f"masked_image_{timestamp}_{i}.jpg"
                cv2.imwrite(masked_image_filename, result['masked_image'])
                result['masked_image_path'] = masked_image_filename
                
            # Sonuçları txt dosyasına yaz
            with open(filename, 'w', encoding='utf-8') as f:
                for result in self.results_list:
                    f.write("=== Analiz Sonucu ===\n")
                    f.write(f"Orijinal Görüntü Dosyası: {result['image_path']}\n")
                    f.write(f"İşlenmiş Görüntü Dosyası: {result['masked_image_path']}\n")
                    f.write(f"Tahmin: {result['prediction']}\n")
                    f.write(f"Konum: X={result['position']['x']:.2f}, "
                            f"Y={result['position']['y']:.2f}, "
                            f"Z={result['position']['z']:.2f}\n")
                    f.write(f"Görüntü Alınma Zamanı: {result['image_timestamp']}\n")
                    f.write(f"Sonuç Oluşturma Zamanı: {result['result_timestamp']}\n")
                    f.write("\n")
            
            self.get_logger().info(f"Sonuçlar {filename} dosyasına kaydedildi.")
            
            # UI'ı başlat
            app = QApplication(sys.argv)
            viewer = ResultViewer(self.results_list)
            viewer.show()
            app.exec_()
            
        except Exception as e:
            self.get_logger().error(f"Dosya kaydetme hatası: {e}")

    def extract_features(self, image):
        """ResNet50'nin avgpool katmanından 2048 boyutlu öznitelik çıkarır."""
        processed_image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.resnet_model.conv1(processed_image)
            features = self.resnet_model.bn1(features)
            features = self.resnet_model.relu(features)
            features = self.resnet_model.maxpool(features)
            features = self.resnet_model.layer1(features)
            features = self.resnet_model.layer2(features)
            features = self.resnet_model.layer3(features)
            features = self.resnet_model.layer4(features)
            features = self.resnet_model.avgpool(features)
            features = torch.flatten(features, 1)
        
        return features.cpu().numpy().squeeze()

    def publish_image(self, cv_image):
        """İşlenmiş görüntüyü /processed_image topic'ine yayınla."""
        try:
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.image_publisher.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Görüntü yayınlama hatası: {e}")

class ResultViewer(QMainWindow):
    def __init__(self, results_list):
        super().__init__()
        self.results_list = results_list
        self.current_index = 0
        self.initUI()
        self.loadCurrentResult()

    def initUI(self):
        self.setWindowTitle('Drone Görüntü Analiz Sonuçları')
        self.setGeometry(100, 100, 1200, 600)  # Genişliği artırdım

        # Ana widget ve layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)  # Ana layout'u yatay yaptım

        # Sol taraf için layout (mevcut görüntü ve bilgiler)
        left_layout = QVBoxLayout()
        
        # Görüntü gösterme alanı
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px solid #cccccc; }")
        left_layout.addWidget(self.image_label)

        # Sonuç bilgileri
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        info_frame.setStyleSheet("QFrame { background-color: #ffffff; padding: 10px; }")
        info_layout = QVBoxLayout(info_frame)
        
        self.prediction_label = QLabel()
        self.position_label = QLabel()
        self.timestamp_label = QLabel()
        
        for label in [self.prediction_label, self.position_label, self.timestamp_label]:
            label.setStyleSheet("QLabel { font-size: 12pt; padding: 5px; }")
            info_layout.addWidget(label)
        
        left_layout.addWidget(info_frame)

        # Navigasyon butonları
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton('Önceki')
        self.prev_button.clicked.connect(self.showPrevious)
        self.prev_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.next_button = QPushButton('Sonraki')
        self.next_button.clicked.connect(self.showNext)
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.counter_label = QLabel()
        self.counter_label.setStyleSheet("QLabel { font-size: 12pt; }")
        
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.counter_label)
        nav_layout.addWidget(self.next_button)
        left_layout.addLayout(nav_layout)

        # Sol layout'u ana layout'a ekle
        layout.addLayout(left_layout)

        # Sağ taraf için scatter plot
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Matplotlib figure oluştur
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        
        layout.addWidget(right_widget)
        
        # Scatter plot'u oluştur
        self.create_scatter_plot()

    def create_scatter_plot(self):
        # Tahmin sınıflarını açıklamalı metinlere eşleyen harita
        color_map = {
            "Sağlıklı (Healthy)": 'green',
            "Sarı Pas (Leaf Rust)": 'red',
            "Rastık (Smut)": 'yellow',
            "Kara Pas (Stem Rust)": 'blue'
        }

        # Verileri hazırla
        x_coords = [result['position']['x'] for result in self.results_list]
        y_coords = [result['position']['y'] for result in self.results_list]
        predictions = [result['prediction'] for result in self.results_list]
        
        # Renkleri hazırla
        colors = [color_map.get(pred, 'gray') for pred in predictions]

        # Plot oluştur
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        scatter = ax.scatter(x_coords, y_coords, c=colors)

        # Etiketleri güncelle
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, label=label, 
                                    markersize=10) 
                        for label, color in color_map.items()]
        ax.legend(handles=legend_elements)

        ax.set_xlabel('X Koordinatı')
        ax.set_ylabel('Y Koordinatı')
        ax.set_title('Drone Konumları ve Sınıflandırma Sonuçları')
        ax.grid(True)

        # Güncel noktayı vurgula
        if self.results_list:
            current_x = self.results_list[self.current_index]['position']['x']
            current_y = self.results_list[self.current_index]['position']['y']
            ax.plot(current_x, current_y, 'ko', markersize=15, fillstyle='none')

        self.canvas.draw()

    def loadCurrentResult(self):
        if not self.results_list or self.current_index >= len(self.results_list):
            return

        result = self.results_list[self.current_index]
        
        # İşlenmiş görüntüyü yükle
        try:
            img = result['masked_image']
            if img is not None:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_img.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                scaled_pixmap = QPixmap.fromImage(qt_img).scaled(
                    self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            self.image_label.setText("İşlenmiş görüntü yüklenemedi")

        # Bilgileri güncelle
        self.prediction_label.setText(f"Tahmin: {result['prediction']}")
        self.position_label.setText(
            f"Konum: X={result['position']['x']:.2f}, "
            f"Y={result['position']['y']:.2f}, "
            f"Z={result['position']['z']:.2f}"
        )
        self.timestamp_label.setText(
            f"Görüntü Zamanı: {result['image_timestamp']}\n"
            f"İşleme Zamanı: {result['result_timestamp']}"
        )

        # Sayaç güncelle
        self.counter_label.setText(f"{self.current_index + 1} / {len(self.results_list)}")

        # Buton durumlarını güncelle
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.results_list) - 1)

        # Scatter plot'u güncelle
        self.create_scatter_plot()

    def showPrevious(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.loadCurrentResult()

    def showNext(self):
        if self.current_index < len(self.results_list) - 1:
            self.current_index += 1
            self.loadCurrentResult()
            
def main(args=None):
    rclpy.init(args=args)
    ai_model_node = AIModelNode()
    try:
        rclpy.spin(ai_model_node)
    except KeyboardInterrupt:
        ai_model_node.save_results_to_file()  # Save results when shutting down
        pass
    finally:
        ai_model_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()