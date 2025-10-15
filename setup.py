from setuptools import setup, find_packages

setup(
    name="pyetc_web",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["flask"],
    description="Web app pyetc",
    author="matteoferro",
    author_email="",
    url="https://github.com/ferromatteo/pyetc_web",
)