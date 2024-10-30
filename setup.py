from setuptools import setup

setup(
    name='tpu_pod_launcher',
    version='1.0.0',
    description='Launch experiments on TPU pods in python.',
    url='https://github.com/Sea-Snell/tpu_pod_launcher/tree/main',
    author='Charlie Snell',
    install_requires=['tyro'],
    py_modules=["tpu_pod_launcher"],
    license='LICENCE',
)
