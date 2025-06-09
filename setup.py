from setuptools import setup, find_packages

setup(
    name="sstac",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.28.0',
        'pandas>=1.5.3',
        'plotly>=5.13.0',
        'numpy>=1.23.5',
        'supabase>=1.0.3',
        'python-dotenv>=0.21.1',
        'pint>=0.20.1',
        'st-supabase-connection>=1.0.2',
        'scipy>=1.9.3'
    ],
    python_requires='>=3.10',
)
