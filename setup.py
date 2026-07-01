from setuptools import setup

setup(
    name='AURAMAXXING',
    version='0.1.0',
    packages=['base', 'growth', 'helpers', 'interfaces', 'saturation', 'rules', 'runner', 'stats', 'dft'],
    python_requires='>=3.10',
    install_requires=['numpy', 'scipy', 'networkx', 'ase', 'tqdm', 'pyyaml'],
    entry_points={'console_scripts': [
        'auramaxxing = runner.__main__:main',
        'auramaxxing-dft = dft.cli:main',
    ]},
    license='MIT',
    author='Alexander Kolganov',
    author_email='alexandr.a.kolganov@gmail.com',
    description='This repository provides tools to grow and saturate mixed Si/Al (at the moment) amorphous oxide structures using ASE and LAMMPS (for growth) and ASE + MACE (for saturation).'
)
