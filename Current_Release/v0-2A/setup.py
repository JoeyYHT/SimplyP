# -*- coding: utf-8 -*-
""" setuptools for simplyP
"""
from setuptools import setup
from codecs import open

setup(name='simplyP',
      version='0.2',
      packages=['simplyP',],
      author='Leah Jackson-Blake',
      author_email='leah.jackson-blake@niva.no',
      url='https://github.com/NIVANorge/niva_datasci_toolkit',
      description='A parsimonious, semi-distributed and dynamic phosphorus model, implemented in Python.',
      long_description=open('../../README.md').read(),
      classifiers=['Development Status :: 3 - Alpha',
                   'License :: OSI Approved :: MIT License',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.6'],
      keywords='phosphorus modelling water quality')
