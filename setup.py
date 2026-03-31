import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="micrograd-extended",
    version="0.1.0",
    author="Finn Reimann",
    author_email="finnrmnn@gmail.com",
    description="Extended version of Andrej Karpathy's micograd.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/finnrmn/micrograd-extended",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requiers=">=3.6",
)
