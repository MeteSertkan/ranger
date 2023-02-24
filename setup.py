# thanks to https://github.com/terrierteam/ir_measures/blob/main/setup.py !! for the setup.py file blueprint!
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

def read(rel_path):
    import os
    import codecs
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    import os
    suffix = os.environ["VERSION_SUFFIX" ] if "VERSION_SUFFIX" in os.environ else ""
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1] + suffix
    else:
        raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="ranger",
    version=get_version("ranger/__init__.py"),
    author="Mete Sertkan, Sophia Althammer, Sebastian Hofstätter",
    description="Ranger is an effect-size meta analysis library creating beautiful forest plots",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data = True,
    packages=setuptools.find_packages(include=['ranger', 'ranger.*']),
    install_requires=list(open('requirements.txt')),
    classifiers=[],
    python_requires='>=3.6',
    package_data={
        'ranger': ['LICENSE', 'requirements.txt'],
    },
)