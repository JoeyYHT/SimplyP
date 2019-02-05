# -*- coding: utf-8 -*-
""" setuptools for SimplyP.
"""
from setuptools import setup
from codecs import open

# Get the long description from the README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(name='simply_p',
      version='0.1',
      description='A parsimonious, semi-distributed and dynamic phosphorus model, implemented in Python.',
      long_description=long_description,
      url='https://github.com/LeahJB/SimplyP',
      author='Leah Jackson-Blake',
      author_email='leah.jackson-blake@niva.no',
      license='MIT',
      classifiers=['Development Status :: 3 - Alpha',
                   'License :: OSI Approved :: MIT License',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
				   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.6'],
      keywords='phosphorus modelling water quality',
      py_modules=['simply_p'],
      install_requires=['matplotlib==2.0.2',
                        'pandas==0.22.0',
                        'seaborn==0.8.1',
                        'numpy==1.14.2',
                        'scipy==1.0.1',
                        'xlrd==1.1.0'])
