from setuptools import find_namespace_packages
from setuptools import setup
from access_fiu.about import __version__, __license__

setup(
    name="access-fiu",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: Commercial License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
    ],
    author="ConvexHull Technology Private Limited",
    author_email="connect@accessai.co",
    maintainer="Majeed Khan",
    maintainer_email="majeed.khan@accessai.co",
    version=__version__,
    packages=find_namespace_packages(exclude=["sample", "tests"]),
    license=__license__,
    long_description=open("README.md").read(),
    install_requires=[
        "opencv-python==4.1.0.25",
        "keras==2.2.4",
        "mtcnn==0.0.9",
        "tensorflow==1.12.3",
        "pillow==6.0.0",
        "scikit-learn==0.21.2"
    ],
)
