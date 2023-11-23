from setuptools import setup, find_packages

# local install of library: pip3 install -e .

setup(
    name='pybondmachine',
    version='1.0.6',
    packages=find_packages(),
    install_requires=[
        # Add any dependencies required by your package
    ],
    entry_points={
        'console_scripts': [
            # If your package has any command line scripts, define them here
        ],
    },
    author='Giulio Bianchini',
    author_email='gibianch@pg.infn.it',
    description='pybondmachine is the Python library designed to streamline the development of FPGA accelerators through the use of the BondMachine framework.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    license='',
)
