from setuptools import find_namespace_packages
from setuptools import setup
from access_face_vision.about import __version__, __license__

setup(
    name="access-face-vision",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries"
    ],
    author="ConvexHull Technology Private Limited",
    author_email="support@accessai.co",
    maintainer="Majeed Khan",
    maintainer_email="majeed.khan@accessai.co",
    version=__version__,
    packages=find_namespace_packages(exclude=["samples", "tests.*", "tests", "*.models"]),
    license=__license__,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://accessai.co/access-face-vision.html",
    install_requires=[
        "opencv-python==4.1.0.25",
        "keras==2.2.4",
        "mtcnn==0.0.9",
        "tensorflow==1.12.3",
        "pillow==6.0.0",
        "scikit-learn==0.21.2",
        "pypiwin32==223",
        "pymongo==3.8.0",
        "sanic==19.6.2",
        "sanic_cors==0.9.8.post1",
        "wget==3.2"
    ],
)
