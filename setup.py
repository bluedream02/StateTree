from setuptools import setup, find_packages

setup(
    name="statetree",
    version="0.1.0",
    description="StateTree + vendored ROLL training core",
    packages=find_packages(include=["statetree", "statetree.*", "roll", "roll.*"]),
    python_requires=">=3.10",
    install_requires=[],
)
