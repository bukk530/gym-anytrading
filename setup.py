from setuptools import setup, find_packages

setup(
    name='gym_manytrading',
    version='1.2.0',
    packages=find_packages(),

    author='AminHP',
    author_email='mdan.hagh@gmail.com',

    install_requires=[
        'gym>=0.12.5',
        'numpy>=1.16.4',
        'pandas>=0.24.2',
        'matplotlib>=3.1.1'
    ],

    package_data={
        'gym_manytrading': ['datasets/data/*']
    }
)
