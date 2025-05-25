#!/usr/bin/env python3

import aws_cdk as cdk
from zorkgpt_viewer_stack import ZorkGPTViewerStack

app = cdk.App()

# Create the ZorkGPT Viewer stack
ZorkGPTViewerStack(
    app,
    "ZorkGPTViewerStack",
    description="Infrastructure for ZorkGPT Live Viewer - S3 bucket and CloudFront distribution",
    env=cdk.Environment(
        # Use default account and region from AWS CLI/environment
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region"),
    ),
)

app.synth()
