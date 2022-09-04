from setuptools import find_namespace_packages, setup


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


setup(
    name="jass_gym",
    version="13.0",
    description="Jass rllib multiagent env",
    url="tbd",
    packages=find_namespace_packages(),
    install_requires=["wheel"] + parse_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "sjgym = jass_gym.__main__:main",
        ],
    },
)
