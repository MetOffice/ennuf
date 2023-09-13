#  (C) Crown Copyright, Met Office, 2023.


from setuptools import setup, find_packages

setup(
    name='ennuf',
    packages=find_packages(include=['src', 'ennuf.*']),
)

