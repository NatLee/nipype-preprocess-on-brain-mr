# syntax=docker/dockerfile:1
#
# One image with exactly the tools the pipeline runs, and nothing else:
#   FSL  fslreorient2std, flirt, bet, fast   <- FSL's conda channel, 5 packages
#   ANTs N4BiasFieldCorrection, Atropos      <- two binaries from the official release
#   ART  acpcdetect                          <- bundled in ./utils
#   FSL's MNI152 T1 template (1 mm and 2 mm) <- TemplateFlow mirror, checked by SHA-256
#
# FSL is free for non-commercial use only (see its licence): build this image
# yourself and do not publish it to a registry.
#
#   docker build -t nipype-preprocess-on-brain-mr .
#   docker run --rm --memory 8g -v "$PWD/data:/data" -v "$PWD/output:/output" nipype-preprocess-on-brain-mr

ARG PYTHON_VERSION=3.14
ARG MICROMAMBA_VERSION=2.3.3

# ---------------------------------------------------------------- FSL ----
FROM mambaorg/micromamba:${MICROMAMBA_VERSION} AS fsl
ARG FSL_CONDA_CHANNEL=https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/
USER root
COPY docker/fsl-packages.txt /tmp/fsl-packages.txt
# Headers, sources, docs and static libraries are only needed to compile against FSL.
RUN micromamba create -y -p /opt/fsl -c "${FSL_CONDA_CHANNEL}" -c conda-forge --override-channels \
        --file /tmp/fsl-packages.txt \
 && micromamba clean -afy \
 && rm -rf /opt/fsl/include /opt/fsl/src /opt/fsl/share/doc /opt/fsl/share/man \
 && find /opt/fsl -name '*.a' -delete

# ------------------------------------------------- ANTs and templates ----
FROM python:${PYTHON_VERSION}-slim AS downloads
ARG ANTS_VERSION=2.6.5
ARG ANTS_BUILD=ubuntu-24.04-X64-gcc
ARG TEMPLATE_URL=https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends ca-certificates curl unzip \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /downloads
# The release zip holds every ANTs program (~750 MB); keep the two the pipeline uses.
RUN curl -fsSL -o ants.zip "https://github.com/ANTsX/ANTs/releases/download/v${ANTS_VERSION}/ants-${ANTS_VERSION}-${ANTS_BUILD}.zip" \
 && mkdir -p /opt/ants/bin \
 && unzip -q -j ants.zip "*/bin/N4BiasFieldCorrection" "*/bin/Atropos" -d /opt/ants/bin \
 && rm ants.zip
COPY docker/templates.sha256 ./
RUN curl -fsSL -o MNI152_T1_1mm.nii.gz "${TEMPLATE_URL}/tpl-MNI152NLin6Asym_res-01_T1w.nii.gz" \
 && curl -fsSL -o MNI152_T1_2mm.nii.gz "${TEMPLATE_URL}/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz" \
 && sha256sum -c templates.sha256

# --------------------------------------------- base (everything but tests) ----
FROM python:${PYTHON_VERSION}-slim AS base
# graphviz draws the workflow graph; acpcdetect links against BLAS/LAPACK and
# calls netpbm's pnmtopng for its landmark previews; FSL's bet script does its
# arithmetic with bc and dc.
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends bc dc graphviz libblas3 liblapack3 netpbm \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY --from=fsl /opt/fsl /opt/fsl
COPY --from=downloads /opt/ants /opt/ants
COPY --from=downloads /downloads/MNI152_T1_1mm.nii.gz /downloads/MNI152_T1_2mm.nii.gz /opt/fsl/data/standard/
# acpcdetect and the three files it reads from $ARTHOME. Without PILbrain.nii it
# still exits 0 but writes a wrongly oriented scan, so all three are required.
COPY utils/acpcdetect_v2.1_LinuxCentOS6.7/bin/acpcdetect /utils/acpcdetect_v2.1_LinuxCentOS6.7/bin/
COPY utils/acpcdetect_v2.1_LinuxCentOS6.7/T1acpc.mdl utils/acpcdetect_v2.1_LinuxCentOS6.7/orion.mdl \
     utils/acpcdetect_v2.1_LinuxCentOS6.7/PILbrain.nii /utils/acpcdetect_v2.1_LinuxCentOS6.7/

# FSL's bin goes last on PATH so its own Python never shadows the pipeline's.
# HOME and MPLCONFIGDIR point at /tmp so the image also runs as a non-root
# user (docker run --user "$(id -u):$(id -g)" keeps output files yours).
ENV FSLDIR=/opt/fsl \
    FSLOUTPUTTYPE=NIFTI_GZ \
    ARTHOME=/utils/acpcdetect_v2.1_LinuxCentOS6.7 \
    PATH=/opt/ants/bin:${PATH}:/opt/fsl/bin \
    NIPYPE_NO_ET=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY src /src
WORKDIR /src

# --------------------------------------------- test (CI and local checks) ----
FROM base AS test
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt
COPY pytest.ini /pytest.ini
COPY tests /tests
WORKDIR /
CMD ["python", "-m", "pytest", "/tests"]

# ------------------------------------------------------------ runtime ----
# Last on purpose: a plain "docker build" builds this stage.
FROM base AS runtime
CMD ["python", "workflow.py"]
