from setuptools import setup,find_packages

def get_requirements(file_path):
    with open(file_path) as file_obj:
        return [line.strip() for line in file_obj if line.strip() and not line.startswith("#") and not line.startswith("-e")]
 

setup(
    name='mlgenericproject',
    version='0.0.1',
    author='morty',
    author_email = 'enugulamaruthi@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)