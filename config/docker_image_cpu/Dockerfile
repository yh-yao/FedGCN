FROM rayproject/ray:latest
MAINTAINER JiayuChang
RUN pip3 install -U torch==1.13 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f "https://data.pyg.org/whl/torch-1.13.0+cpu.html"
