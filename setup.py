from setuptools import setup, find_packages
import sys

if sys.version.startswith('2.7'):
    pathlib = 'pathlib2>=1.0.1'
elif sys.version.startswith('3.5') or sys.version.startswith('3.6'):
    pathlib = 'pathlib>=1.0.1'
else:
    raise Exception('Only Python 2.7, 3.5 and 3.6 are supported')

print('Please install OpenAI Baselines (commit 8e56dd) and requirement.txt')
if not (sys.version.startswith('3.5') or sys.version.startswith('3.6')):
    raise Exception('Only Python 3.5 and 3.6 are supported')

setup(name='deep_rl',
      packages=[package for package in find_packages()
                if package.startswith('deep_rl')],
      install_requires=['torch>=0.4.0',
          'torchvision>=0.2.1',
          'gym>=0.10.5',
          'atari-py>=0.1.1',
          'opencv-python>=3.4.0.12',
          'tensorboardX==1.1',
          'scikit-image>=0.13.1',
          'tqdm>=4.23.0',
          'pandas>=0.22.0',
          'seaborn>=0.8.1',
pathlib],
      description="Modularized Implementation of Deep RL Algorithms",
      author="Shangtong Zhang",
      url='https://github.com/ShangtongZhang/DeepRL',
      author_email="zhangshangtong.cpp@gmail.com",
      version="0.5")