import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gmnr",
    version="1.0.0",
    description="Generative MPI Neural Radiance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VIROBO-15/GMNR",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    py_modules=["gmnr"],
)
