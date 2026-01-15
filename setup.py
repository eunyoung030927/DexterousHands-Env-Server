"""Installation script for the 'isaacgymenvs' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))


def parse_requirements(filename):
    with open(os.path.join(root_dir, filename), "r") as f:
        requirements = f.read().splitlines()
        # 빈 줄이나 주석(#) 제거
        requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]
        # -e git+... 같은 링크 형태 제거 (setup.py 오류 방지)
        requirements = [r for r in requirements if not r.startswith("-e") and not r.startswith("git+")]
    return requirements

# requirements.txt 파일이 있으면 읽어오고, 없으면 기본값 사용
req_file = "requirements.txt"
if os.path.exists(os.path.join(root_dir, req_file)):
    INSTALL_REQUIRES = parse_requirements(req_file)
else:
    # 비상용 기본값 (requirements.txt가 없을 때)
    INSTALL_REQUIRES = [
        "gym",
        "matplotlib==3.5.1",
        "tqdm",
        "ipdb",
        "numpy==1.23.0",
        "rl-games==1.5.2",
    ]

# Installation operation
setup(
    name="bidexhands",
    author="cypypccpy",
    version="0.1.2",
    description="Benchmark environments for Dexterous Hand in NVIDIA IsaacGym.",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,  # 여기서 위에서 읽은 리스트를 사용
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7"],
    zip_safe=False,
)
