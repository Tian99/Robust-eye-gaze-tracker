from debian:bullseye-slim
ENV LANG=C.UTF-8 ENABLE_CONTRIB=1 ENABLE_HEADLESS=1 
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip python3-scipy python3-pyqtgraph \
	&& rm -rf /var/lib/apt/lists/*

# get rest of the depends
WORKDIR /tracker
CMD ["python3", "-m", "pytest", "test/"]
COPY requirements.txt.min /
RUN python3 -m pip install -r /requirements.txt.min
COPY . /tracker/


# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#       python3-pip python3-numpy \
#       gcc g++ cmake git \
#       build-essential \
# 	&& rm -rf /var/lib/apt/lists/*
# # cmake+pip doesn't work? and building elsewhere requires copying most of the same depends
# # RUN python3 -m pip install --no-binary :all: opencv-python
# # COPY --from=opencv_contrib_headless /opencv-python/opencv_contrib_python_headless.whl .
# 
# # make opencv_contrib_python_headless
# # latest as of 20210226 (committed 20201228)
# RUN git clone --recursive https://github.com/skvark/opencv-python.git 
# RUN \
#   cd opencv-python  && \
#   git checkout fd4e6040af94d && \
#   python3 -m pip wheel . --verbose
# # TODO above and below can be merged when it works
# RUN find -iname '*whl' -exec mv {} opencv_contrib_python_headless.whl \;
# RUN python3 -m pip install -e opencv_contrib_python_headless.whl
