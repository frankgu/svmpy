from setuptools import setup

setup(name='svmpy',
      version='1',
      long_description=open("README.rst").read(),
      url='https://github.com/frankgu/svmpy',
      author='Dongfeng Gu',
      author_email='gudongfeng@outlook.com',
      license='MIT',
      packages=['svmpy'],
      install_requires=[
          'argh',
          'numpy',
          'cvxopt'
      ],
      zip_safe=False)
