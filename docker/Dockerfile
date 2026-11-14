# One image both builds the PCBWorld engine against KiCad 10.0.7 and runs PCBRouteAgent on it.
# The KiCad packages follow Build-Depends of the noble source package in ppa:kicad/kicad-10.0-releases,
# without the documentation tools; WebView, Poppler, SPNAV and X11 headers are the ones the engine's
# KiCad 10 port lists for Linux.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config git rsync ca-certificates curl \
      python3-dev python3-venv swig \
      libbz2-dev libcairo2-dev libglu1-mesa-dev libgl1-mesa-dev libglew-dev libx11-dev mesa-common-dev \
      libwxgtk3.2-dev libwxgtk-webview3.2-dev libgtk-3-dev \
      libssl-dev libzstd-dev zlib1g-dev libgit2-dev libsecret-1-dev libcurl4-openssl-dev \
      libboost-all-dev libglm-dev libngspice0-dev unixodbc-dev libspnav-dev \
      libpoppler-dev libpoppler-cpp-dev libpoppler-glib-dev \
      libnng-dev libprotobuf-dev protobuf-compiler shared-mime-info \
      libocct-modeling-algorithms-dev libocct-modeling-data-dev libocct-data-exchange-dev \
      libocct-visualization-dev libocct-foundation-dev libocct-ocaf-dev \
      libcairo2 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Only what PCBRouteAgent, pcb_world.engine and eval import at run time; PCBWorld's own lock adds
# the training stack (torch, vllm, ray), which routing does not use.
RUN python3 -m venv /opt/venv && /opt/venv/bin/python -m pip install --no-cache-dir \
      pybind11==3.0.3 numpy==2.2.6 pyyaml==6.0.3 filelock==3.25.2 cloudpickle==3.1.2 psutil==7.2.2 \
      tqdm==4.67.3 gymnasium==1.2.3 pypdfium2==5.12.1 pillow==12.1.1 matplotlib==3.10.8 pandas==3.0.1 \
      shapely==2.1.2 "scipy>=1.13,<2" "jsonschema>=4.23,<5" anthropic==0.111.0 "cairosvg>=2.9,<3"
ENV PATH=/opt/venv/bin:$PATH
