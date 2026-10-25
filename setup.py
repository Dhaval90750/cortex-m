from setuptools import setup, find_packages

setup(
    name="cortex-m",
    version="2.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "rich>=13.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    package_data={
        "": ["*.html", "*.css", "*.js", "*.json"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "cortex = cortex_m.cli:cortex_unified_main",
            "cortex-studio = cortex_m.cli:cortex_studio_main",
            "cortexstudio = cortex_m.cli:cortex_studio_main",
            "cortexcode = cortex_m.cli:cortex_code_main",
            "cortex-code = cortex_m.cli:cortex_code_main",
            "cortex-substrate = cortex_m.cli:cortex_substrate_main",
            "cortexlm = cortex_m.cli:cortex_lm_main",
            "cortex-lm = cortex_m.cli:cortex_lm_main",
        ],
    },
)
