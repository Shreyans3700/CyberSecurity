"""
Setup Configuration for CyberSecurity Project

This module is responsible for packaging and distributing the CyberSecurity
project using setuptools. It defines project metadata and dynamically reads
dependencies from a requirements.txt file.

Key Functionalities:
- Uses setuptools to package the project.
- Automatically discovers all packages using find_packages().
- Reads and parses dependencies from requirements.txt using a custom function.
- Filters out comments and unnecessary lines from the requirements file.

Functions:
- get_requirements(file_path: str) -> List[str]:
    Reads the specified requirements file and returns a cleaned list of
    dependencies. It ignores commented lines and handles missing file errors.

Setup Metadata:
- name: Project name (CyberSecurity)
- version: Initial version (0.0.1)
- author: Shreyansh Pandey
- author_email: pandeyshreyansh46@gmail.com
- packages: Automatically discovered Python packages
- install_requires: List of dependencies required for the project

Error Handling:
- Raises FileNotFoundError if the requirements file is not found.
- Raises generic exceptions for unexpected issues during file parsing.

Usage:
Run the following command to install the package locally:
    pip install .

Or for development mode:
    pip install -e .

This setup script ensures that all dependencies are installed and the project
is packaged correctly for distribution or deployment.
"""

from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """
    Reads a requirements file and returns a clean list of dependencies.
    Ignores comments, empty lines, and editable installs (-e .).
    """
    try:
        requirements = []

        with open(file_path) as file:
            lines = file.readlines()

            for line in lines:
                requirement = line.strip()

                # Skip empty lines, comments, and editable installs
                if (
                    requirement
                    and not requirement.startswith("#")
                    and not requirement.startswith("-e")
                ):
                    requirements.append(requirement)

        return requirements

    except FileNotFoundError:
        raise FileNotFoundError(f"Requirements file not found at path: {file_path}")
    except Exception as e:
        raise Exception(f"Error while reading requirements: {e}")


setup(
    name="CyberSecurity",
    version="0.0.1",
    packages=find_packages(),
    author="Shreyansh Pandey",
    author_email="pandeyshreyansh46@gmail.com",
    install_requires=get_requirements("requirements.txt"),
)
