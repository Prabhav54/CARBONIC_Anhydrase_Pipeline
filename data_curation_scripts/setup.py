from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    This function reads requirements.txt and returns a list of libraries.
    It automatically removes the '-e .' flag so setup() doesn't fail.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove newlines (\n) from each line
        requirements = [req.replace("\n", "") for req in requirements]
        
        # Remove '-e .' if present (it's for pip, not setuptools)
        if "-e ." in requirements:
            requirements.remove("-e .")
    
    return requirements

setup(
    name='Carbonic_ML_Pipeline',
    version='0.0.1',
    author='Prabhav Khare',  
    author_email='prabhavkhare54@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)