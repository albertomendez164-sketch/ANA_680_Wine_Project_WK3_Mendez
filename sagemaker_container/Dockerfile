FROM python:3.12-slim
RUN pip install --no-cache-dir boto3 sagemaker-training pandas==2.3.3 numpy==2.2.6 scikit-learn==1.7.2 joblib==1.5.2
COPY train.py /opt/program/train.py
ENV SAGEMAKER_PROGRAM=train.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/program
WORKDIR /opt/program
ENTRYPOINT ["python", "/opt/program/train.py"]
