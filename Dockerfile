FROM python:3.8
WORKDIR /workspace
ADD . /workspace
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python" , "/workspace/app.py" , "gunicorn", "--config", "gunicorn-cfg.py" ]
RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace
