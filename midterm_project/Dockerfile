# =============================================================================
# Dockerfile for the rain prediction web service (predict.py + weather_forecast.bin)
#
# Key ideas:
#   - Dockerfile = the recipe (a list of steps, run from top to bottom)
#   - Image      = the finished package, built from the recipe with `docker build`
#   - Container  = a running copy of the image, started with `docker run`
#   A container is like a small, separate Linux computer: it can only see files
#   you COPY into it, and it can only be reached through ports you publish.
#
# Build the image (run from the project folder; "." = use the Dockerfile here):
#   docker build -t rain-prediction .
#
# Run a container (-p laptop_port:container_port connects the two ports,
#   --rm deletes the container when it stops, -it shows logs and lets Ctrl+C stop it):
#   docker run -it --rm -p 9696:9696 rain-prediction
#
# Test it:
#   python test.py                      (sends a sample record to /predict)
#   http://localhost:9696/docs          (FastAPI's interactive test page)
#   Note: open "localhost" in the browser, not "0.0.0.0".
#
# Reusing this file for another project: usually you only change
#   1) the Python version in FROM,  2) the files in the second COPY,
#   3) the port in EXPOSE and CMD,  4) the "module:app" name in CMD.
# =============================================================================


# FROM = the starting image that everything else is built on top of.
# python:3.11-slim is a small Debian Linux with Python 3.11 already installed.
# The version must match .python-version (3.11); "slim" leaves out extra tools we don't need.
FROM python:3.11-slim

# RUN = run a shell command while the image is being BUILT (not when the container starts).
# The base image only has pip, so we use it to install uv, the same tool we use on the laptop.
RUN pip install uv

# WORKDIR = create this folder inside the image and work from it for all later steps
# (like `mkdir /app && cd /app`). Relative paths such as "./" below now mean /app.
WORKDIR /app

# COPY <files on the laptop> <destination in the image>
# Copy ONLY the dependency files first. Docker caches every step as a "layer" and reuses
# it if its inputs haven't changed. Because these two files rarely change, the slow
# install step below is cached, and editing predict.py later won't reinstall everything.
COPY pyproject.toml uv.lock ./

# Install the dependencies listed in uv.lock into /app/.venv inside the image.
#   --locked              use the exact versions from uv.lock (same as on the laptop);
#                         fail instead of silently updating if uv.lock is out of date
#   --no-install-project  install only the libraries, not our own src/ package
#                         (the web service doesn't need it, and it isn't copied in)
RUN uv sync --locked --no-install-project

# Now copy the files the service actually needs: the code and the saved model.
# Everything else (CSV, notebooks, backups, .venv) stays out, which keeps the image small.
# These change more often, so they come AFTER the slow install step (see the caching note above).
COPY predict.py weather_forecast.bin ./

# ENV = set an environment variable for the build steps and for the running container.
# Putting the virtual environment's bin folder first on PATH means `python` and `uvicorn`
# come from /app/.venv. It's the same as running `activate` on the laptop.
ENV PATH="/app/.venv/bin:$PATH"

# EXPOSE = record which port the app listens on. It does NOT open the port by itself
# (`docker run -p 9696:9696` does that locally), but Elastic Beanstalk reads this line
# to know where to send incoming web traffic.
EXPOSE 9696

# CMD = the command that runs when the container STARTS (not during the build).
# Written as a JSON list: the program, then each argument as a separate string.
#   predict:app      the `app` object inside predict.py
#   --host 0.0.0.0   accept connections from outside the container
#                    (the default, 127.0.0.1, would only accept connections from inside it)
#   --port 9696      the port to listen on; must match EXPOSE and the -p mapping
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
