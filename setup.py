from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function reads the requirements.txt file and returns a list of libraries to install.
    It removes the '-e .' flag if present to prevent errors.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove newlines (\n) from each line
        requirements = [req.replace("\n", "") for req in requirements]

        # Remove '-e .' if it exists in requirements.txt
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='Carbonic_Anhydrase_Pipeline',
    version='0.0.1',
    author='Prabhav Khare',
    author_email='prabhavkhare54@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)