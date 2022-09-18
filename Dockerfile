FROM python:3.8

ADD config.py .
ADD fit.py .
ADD fit-bel.py .
ADD param.py .
ADD Spectrum.py .
ADD utils.py .
ADD requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "./fit-bel.py"]