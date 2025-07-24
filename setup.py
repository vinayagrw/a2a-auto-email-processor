from setuptools import setup, find_packages

setup(
    name="a2a-mcp-contractor-automation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "aiohttp",
        "httpx",
        "pydantic>=1.10.0",
        "chromadb",
        "a2a-sdk",  # Assuming this is the correct package name for A2A SDK
    ],
    entry_points={
        'console_scripts': [
            'a2a-contractor=main:main',
        ],
    },
)
