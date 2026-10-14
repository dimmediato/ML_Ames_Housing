# Start from AWS's official Lambda image for Python 3.11.
# If your version check printed a different 3.x, change the tag to match.
FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.11

# Install the pinned libraries into the image.
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# Add your handler code.
COPY app.py ${LAMBDA_TASK_ROOT}

# Tell Lambda which function to call: file "app", function "handler".
CMD [ "app.handler" ]