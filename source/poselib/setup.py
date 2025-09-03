"""Installation script for the 'trackerLab' Python package."""

from setuptools import setup

# Package metadata (hardcoded)
PACKAGE_NAME = "poselib"
VERSION = "0.1.0"
AUTHOR = "zaterval"
MAINTAINER = "Tsinghua University"
MAINTAINER_EMAIL = "ziang_zheng@foxmail.com"
DESCRIPTION = "tracking task for isaacLab"
REPOSITORY = ""
KEYWORDS = ["extension", "isaacLab"]

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # "psutil",  # NOTE: Add additional dependencies here
]

# Setup the package installation
setup(
    # Package name and metadata
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    keywords=KEYWORDS,
    url=REPOSITORY,
    license="BSD-3-Clause",
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.0.0",
    ],
    zip_safe=False,
    packages=[PACKAGE_NAME],  # Package directory
)
