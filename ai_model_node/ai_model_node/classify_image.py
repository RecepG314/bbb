#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import joblib
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.timer import Timer

class AIModelNode(Node):
    def __init__(self):
        super().__init__('ai_model_node')

        self.get_logger().info("AI Model Node başlatıldı.")

        # ResNet modelini yükle
        self.device = torch.device('cpu')
        self.resnet_model = models.resnet50(pretrained=True).to(self.device)
        self.resnet_model.eval()  # Modeli değerlendirme moduna al
        self.get_logger().info("ResNet modeli yüklendi.")

        # SVM modelini yükle
        self.svm_model = joblib.load("/root/ros2_ws/src/sjtu_drone/ai_model_node/data/svm_model_resnet50.joblib")
        self.get_logger().info("SVM modeli yüklendi.")

        # Görüntü işleme dönüşümleri
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # ResNet için giriş boyutu
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # CvBridge ile ROS görüntülerini işleyin
        self.bridge = CvBridge()

        # Publisher (Model sonucu ve işlenmiş görüntü)
        self.result_publisher = self.create_publisher(String, '/ai_model/result', 10)
        self.image_publisher = self.create_publisher(Image, '/processed_image', 10)

        # Subscriber (Fotoğraf alımı)
        self.image_subscriber = self.create_subscription(
            Image,
            '/simple_drone/bottom/image_raw',
            self.image_callback,
            10
        )

        # Timer: Görüntü işleme her 5 saniyede bir yapılacak
        self.timer = self.create_timer(5.0, self.timer_callback)
        self.latest_image = None  # En son alınan görüntü

    def image_callback(self, msg):
        """Görüntüyü al ve kaydet"""
        self.latest_image = msg  # Görüntüyü kaydet
        self.get_logger().info('Yeni görüntü alındı ve kaydedildi.')

    def timer_callback(self):
        """5 saniyede bir görüntüyü işle"""
        if self.latest_image is not None:
           # self.get_logger().info('5 saniyede bir tetiklendi, görüntü işleniyor...')

            try:
                # ROS mesajını OpenCV görüntüsüne dönüştür
                cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")

                # Görüntüyü işleyerek öznitelik çıkar
                feature_vector = self.extract_features(cv_image)

                # SVM modelini kullanarak sınıflandır
                prediction = self.svm_model.predict(feature_vector.reshape(1, -1))
                predicted_label = prediction[0]

                # Sonucu yayınla
                result_msg = String()
                result_msg.data = f"Tahmin edilen sınıf: {predicted_label}"
                self.result_publisher.publish(result_msg)
                self.get_logger().info(f"Tahmin sonucu: {predicted_label}")

                # Görüntüyü yayınla
                self.publish_image(cv_image)

            except Exception as e:
                self.get_logger().error(f"Görüntü işleme hatası: {e}")
            finally:
                self.latest_image = None  # Görüntüyü sıfırla

    def extract_features(self, image):
        """ResNet50'nin avgpool katmanından 2048 boyutlu öznitelik çıkarır."""
        processed_image = self.transform(image).unsqueeze(0).to(self.device)  # Görüntüyü işleme sok

        with torch.no_grad():
            # ResNet'in özellik haritasını çıkar
            features = self.resnet_model.conv1(processed_image)
            features = self.resnet_model.bn1(features)
            features = self.resnet_model.relu(features)
            features = self.resnet_model.maxpool(features)
            features = self.resnet_model.layer1(features)
            features = self.resnet_model.layer2(features)
            features = self.resnet_model.layer3(features)
            features = self.resnet_model.layer4(features)
            features = self.resnet_model.avgpool(features)  # 2048 boyutlu özellik vektörü

            # Düzleştir ve numpy'a dönüştür
            features = torch.flatten(features, 1)
        
        return features.cpu().numpy().squeeze()

    def publish_image(self, cv_image):
        """İşlenmiş görüntüyü /processed_image topic'ine yayınla."""
        try:
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")  # OpenCV görüntüsünü ROS Image mesajına dönüştür
            self.image_publisher.publish(img_msg)
            self.get_logger().info("İşlenmiş görüntü /processed_image topic'ine yayınlandı.")
        except Exception as e:
            self.get_logger().error(f"Görüntü yayınlama hatası: {e}")


def main(args=None):
    rclpy.init(args=args)
    ai_model_node = AIModelNode()
    try:
        rclpy.spin(ai_model_node)
    except KeyboardInterrupt:
        pass
    finally:
        ai_model_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
