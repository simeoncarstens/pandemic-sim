import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pandemic-sim",
    version="0.1",
    author="Simeon Carstens",
    author_email="blog@simeon-carstens.com",
    description="Library to simulate, visualize and animate an epidemic using explicit simulation of an agent-based model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simeoncarstens/pandemic-sim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
