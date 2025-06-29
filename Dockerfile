FROM public.ecr.aws/docker/library/python:3.12-slim
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
ENV PORT=8000
WORKDIR /var/task

COPY requirements.txt .
RUN python -m pip install -r requirements.txt 


COPY utils/ ./utils/
COPY main.py ./
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8000
CMD exec uvicorn --port=$PORT main:app --host=0.0.0.0