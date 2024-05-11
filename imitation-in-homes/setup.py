from setuptools import setup, find_packages

setup(
    name='dobbe',
    version='0.1.0',
    description='A brief description of my package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your_email@example.com',
    url='https://github.com/cpaxton/dobbe',
    packages=find_packages(),
    install_requires=[
        #'dependency1>=1.0.0',
        #'dependency2>=2.0.0',
    ],
    classifiers=[
        #'Development Status :: 3 - Alpha',
        #'Intended Audience :: Developers',
        #'License :: OSI Approved :: MIT License',
        #'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.6',
        #'Programming Language :: Python :: 3.7',
        #'Programming Language :: Python :: 3.8',
        #'Programming Language :: Python :: 3.9',
        #'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        #'console_scripts': [
        #    'my-script = my_package.main:main',
        #],
    },
)
