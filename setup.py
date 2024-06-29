import setuptools

setuptools.setup(
    name="llm_tools",
    version="0.0.1",
    author="Géraud Bourdin, Thomas Bourdin",
    author_email="gbourdin@partitech.com, tbourdin@partitech.com",
    description="Une collection d'outils utiles pour l'entraînement de grands modèles de langage.",
    packages=setuptools.find_packages(),
    install_requires=[
        'datasets',
        'mistral_common',
        'transformers'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
