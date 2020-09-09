from setuptools import setup, \
                       find_packages

with open('requirements.txt', 'r') as file:
    requirements = [line for line in file]

with open('README.md', 'r') as file:
    long_description = file.read()

setup(
    name='transformer',
    version='0.1.0',
    description='keras implementation of Transformer as described in "Attention Is All You Need" by Vaswani et. al.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ViktorStagge/transformer',
    author='Viktor Stagge',
    author_email='viktor.stagge@gmail.com',
    license='MIT',
    packages=find_packages(),
    package_data={},
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Data Science :: Transformer',

        # Pick your license as you wish (should match "license" above)
        'License :: MIT',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='ml transformer attention machine learning',
)
