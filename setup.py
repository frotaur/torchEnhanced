from setuptools import setup, find_packages
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='torchenhanced',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    test_suite='tests',
    description='Wrappers for pytorch stuff I use on the daily.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Frotaur',
    author_email='frotaur@hotmail.co.uk',
    url='https://github.com/frotaur/TorchEnhanced',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)