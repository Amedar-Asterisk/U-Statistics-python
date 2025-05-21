from setuptools import setup, find_packages

setup(
    name="U_stats",
    version="0.1",
    author="Zhang Ruiqi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
    ],
)
