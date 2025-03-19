from setuptools import setup, find_packages

setup(
    name="id_mlip",
    version='0.1',
    packages=find_packages(),
    install_requires=[

    ],
    entry_points={
        "console_scripts": [
            "id-mlip = id_mlip:show_message"
        ]
    }
)