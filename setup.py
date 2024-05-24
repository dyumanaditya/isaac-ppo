from setuptools import setup, find_packages

# Read the contents of README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read the contents of requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='isaac_ppo',
    version='1.0.0',
    author='Dyuman Aditya',
    author_email='dyuman.aditya@gmail.com',
    description='Implementation of PPO algorithm using PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dyumanaditya/ppo',
    license='MIT',
    project_urls={
        'Repository': 'https://github.com/dyumanaditya/ppo',
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>3.9',
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True
)
