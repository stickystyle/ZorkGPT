#!/usr/bin/env python3

import os
import aws_cdk as cdk
from zorkgpt_viewer_stack import ZorkGPTViewerStack

app = cdk.App()

# Create the ZorkGPT Viewer stack
ZorkGPTViewerStack(
    app,
    "ZorkGPTViewerStack",
    description="Infrastructure for ZorkGPT Live Viewer - S3 bucket and CloudFront distribution",
    env=cdk.Environment(
        # Use environment variables or CDK context
        account=os.environ.get("CDK_DEFAULT_ACCOUNT")
        or app.node.try_get_context("account"),
        region=os.environ.get("CDK_DEFAULT_REGION")
        or app.node.try_get_context("region"),
    ),
)

app.synth()
