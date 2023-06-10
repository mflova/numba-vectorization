from distutils.core import setup

setup(
    name="Numba vectorization",
    version="1.0",
    python_requires=">=3.8",
    description="Collection of Numba vecotrization-based examples.",
    author="Manuel Floriano",
    install_requires=["numba", "matplotlib", "numpy"],
)
