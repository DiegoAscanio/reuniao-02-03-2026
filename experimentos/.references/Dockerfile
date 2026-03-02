# Container to run CPLEX at linux

# Use the official Ubuntu 22.04 base image as CPLEX doesn't support
# Python 3.12 yet
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    libstdc++6 \
    build-essential \
    python3 \
    python3-pip \
    python-is-python3 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# With pip, allowing system breakage, we'll install numpy, scipy, pandas
# PuLP, ipython, jupyter (lab, notebook, nbclassic), matplotlib, seaborn
# statsmodels, scikit-learn, plotly, cufflinks, openpyxl, xlrd, xlwt,
# xlsxwriter, sympy and networkx
RUN pip3 install \
    numpy \
    scipy \
    pandas \
    pulp \
    ipython \
    jupyterlab \
    notebook \
    nbclassic \
    matplotlib \
    seaborn \
    statsmodels \
    scikit-learn \
    plotly \
    cufflinks \
    openpyxl \
    xlrd \
    xlwt \
    xlsxwriter \
    sympy \
    networkx

# Create a CPLEX user
RUN useradd -m cplex -s /bin/bash
RUN echo "cplex ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN echo "cplex:cplex" | chpasswd

# Set working directory
WORKDIR /home/cplex/workspace

# Copy install_and_set_cplex.sh script to the container
COPY install_cplex.sh /install_cplex.sh
RUN chmod +x /install_cplex.sh

# Expose Jupyter Notebook port
EXPOSE 8888

# Run the CPLEX installation and setup script as an entrypoint
ENTRYPOINT ["/install_cplex.sh"]

# Switch to the CPLEX user
USER cplex

# Set the working directory to the CPLEX user's workspace
WORKDIR /home/cplex/workspace

# Set the default command to bash
CMD ["/bin/bash"]
