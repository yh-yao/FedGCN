FROM rayproject/ray:latest-gpu
MAINTAINER JiayuChang
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
RUN pip install ogb
