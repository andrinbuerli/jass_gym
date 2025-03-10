import os
import subprocess
from pathlib import Path

from setuptools import setup


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

try:
    if 'SKIP_EXTERN' not in os.environ:
        cwd = Path(__file__).parent.resolve()
        subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=cwd)

        subprocess.check_call(["pip", "install", "."], cwd=cwd / "extern" / "jass-kit-py")
        subprocess.check_call(["pip", "install", "."], cwd=cwd / "extern" / "jass-kit-cpp")
        subprocess.check_call(["cmake", "."], cwd=cwd / "extern" / "jass-kit-cpp")
        subprocess.check_call(["make", "install"], cwd=cwd / "extern" / "jass-kit-cpp")
        subprocess.check_call(["pip", "install", "."], cwd=cwd / "extern" / "jass-ml-cpp")
        subprocess.check_call(["pip", "install", "."], cwd=cwd / "extern" / "jass-ml-py")
except Exception as e:
    print("Failed to install extern modules...", e)

setup(
    name="jass_gym",
    version="13.0",
    description="Jass rllib multiagent env",
    url="tbd",
    packages=["jass_gym"],
    install_requires=["wheel"] + parse_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "sjgym = jass_gym.__main__:main",
        ],
    },
)
