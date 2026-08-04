# Troubleshooting Record

Use this file when a deployment or SageMaker training job fails.

## Date and stage

State whether the failure occurred during local testing, Docker build, Heroku
deployment, ECR push, or SageMaker training.

## Exact error

Paste the relevant error or log lines.

## Actions taken

1. State the first diagnostic step.
2. State the configuration or code change.
3. State how the application or training job was retested.

## Final result

State whether the issue was resolved. Include the working application URL,
successful training-job status, or remaining limitation.

## Useful commands

    heroku logs --tail --app YOUR-APP-NAME
    docker logs CONTAINER_ID
    aws sagemaker describe-training-job --training-job-name JOB_NAME
