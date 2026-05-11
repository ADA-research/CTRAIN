from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "torch>=2",
    "torchvision>=0.12.0",
    "numpy>=1.20,<2",
    "tqdm>=4",
    "smac>=2.3",
    "pandas>=2",
    "matplotlib>=3",
    "seaborn",
    "onnxruntime>=1",
    "onnxsim>=0.4.31",
    "onnxoptimizer",
    "skl2onnx",
    "appdirs>=1.4",
    "graphviz>=0.20.3",
    "requests",
    "ninja>=1.10",
    "packaging>=20.0",
    "psutil>=5.9.5",
    "pyyaml>=6.0",
    "sortedcontainers>=2.4",
    "termcolor>=2.3.0",
    "gurobipy>=10",
    "onnx2pytorch @ git+https://github.com/Verified-Intelligence/onnx2pytorch.git@8447c42c3192dad383e5598edc74dddac5706ee2",
    "auto_LiRPA @ git+https://github.com/Verified-Intelligence/auto_LiRPA.git@9d100ec070868440b48d34e2f1dd21b97aab9172",
]

setup(
    name="CTRAIN",
    version="0.4.3",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    package_data={
        "CTRAIN": ["verification_systems/abCROWN/**/*"],
    },
    include_package_data=True,
)
