from setuptools import setup
from setuptools import find_namespace_packages

setup(name='RadarGym',
      version='0.1',
      description='Reinforcement learning environments for radar applications.',
      url='https://github.com/PetteriPulkkinen/RadarGym.git',
      author='Petteri Pulkkinen',
      author_email='petteri.pulkkinen@aalto.fi',
      licence='MIT',
      packages=find_namespace_packages(),
      install_requires=[
            'numpy', 'gym', 'trackingsimpy'
      ],
      zip_safe=False)
