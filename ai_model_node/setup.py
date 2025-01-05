from setuptools import setup

package_name = 'ai_model_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='A ROS2 package for AI image classification',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Sanal ortam Python yorumlayıcısını kullan
            'classify_image = ai_model_node.classify_image:main'
        ],
    },
    python_requires='>=3.10',  # Python sürümünü belirleyin
)
