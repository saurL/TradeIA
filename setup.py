from setuptools import setup, find_packages

setup(
    name='IaFinance',              # Nom du package
    version='0.1.0',                    # Version du package
    packages=find_packages(),           # Inclure tous les sous-packages
    author='Lucas Sauron',                 # Nom de l'auteur
    author_email='sauronlucas@gmail.com',  # Email de l'auteur
    description='Une courte description du package',  # Description du package
    long_description=open('README.md').read(),  # Description longue (extrait du README)
    long_description_content_type='text/markdown',  # Format du long_description
    url='https://github.com/saurL/TradeIA/',        # URL du projet (repository, site web, etc.)
    classifiers=[                       # Classificateurs pour le package
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3',            # Version de Python requise
)