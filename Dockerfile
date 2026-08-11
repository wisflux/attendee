# Application image for Attendee. Builds on the prebuilt base image
# (Dockerfile.base -> wisfluxp/attendee-base), which carries all the heavy,
# slow-changing system dependencies (build toolchain, OpenCV, X11, audio,
# GStreamer, Chrome + ChromeDriver, tini, python). This image only adds the
# Python dependencies and the application code, so it builds in seconds and
# runs no apt.
#
# When the base image's system dependencies change, rebuild and push
# wisfluxp/attendee-base first (see Dockerfile.base), then rebuild this image.
FROM --platform=linux/amd64 wisfluxp/attendee-base:latest AS build

SHELL ["/bin/bash", "-c"]

ENV project=attendee
ENV cwd=/$project

WORKDIR $cwd

# Python dependencies. Copy requirements first so this layer only rebuilds when
# they change. pyav must be compiled from source against the base image's
# libavdevice-dev (not the prebuilt wheel) for webpage streaming to work.
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip install --no-binary av --force-reinstall --no-deps "av==12.0.0"

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash app

COPY --chown=app:app --chmod=0755 entrypoint.sh /usr/local/bin/entrypoint.sh
COPY --chown=app:app . .

RUN mkdir -p "$cwd/staticfiles" && chown -R app:app "$cwd/staticfiles"

RUN mkdir -p /etc/opt/chrome/policies/managed \
  && ln -s /tmp/attendee-chrome-policies.json /etc/opt/chrome/policies/managed/attendee-chrome-policies.json

USER app

ENTRYPOINT ["/tini","--","/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
