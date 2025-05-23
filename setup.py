from setuptools import setup, find_packages

setup(
    name='test-obj-det',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'ultralytics',
        'scikit-learn',
        'pygments',
        'IPython',
        'numpy',
        'opencv-python'
    ],
)