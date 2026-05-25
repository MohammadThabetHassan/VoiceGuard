---
name: aws-launch-gpu
description: Generate (DO NOT EXECUTE) a GPU launch script for training. Presents cost estimate and waits for user confirmation.
disable-model-invocation: true
argument-hint: "<task-name> <estimated-hours>"
---

## Instructions

Generate `scripts/aws/launch_$TASK.sh` with the template below.
**Do NOT execute it.** Present the script and cost estimate to the user. STOP.

```bash
#!/usr/bin/env bash
# Cost estimate: g5.xlarge Spot ~$0.90/hr × <hours> hr = ~$<total>
# Run time estimate: <hours> hours
# ⚠️  This script will launch a GPU instance and incur AWS charges.
# Review carefully. Cancel within 10 seconds by pressing Ctrl+C.

set -euo pipefail

TASK="<task-name>"
REGION="ap-southeast-1"
S3_BUCKET="voiceguard-mt-2026"
INSTANCE_TYPE="g5.xlarge"
AMI_ID=""  # TODO: fill in Deep Learning AMI for ap-southeast-1
KEY_NAME=""  # TODO: fill in your key pair name
SECURITY_GROUP=""  # TODO: fill in your security group

echo "Launching $INSTANCE_TYPE Spot for task: $TASK"
echo "Estimated cost: ~\$<total> | Duration: ~<hours>h"
echo "Checkpoints → s3://$S3_BUCKET/checkpoints/$TASK/"
sleep 10  # Cancel window

INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SECURITY_GROUP" \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=voiceguard-$TASK},{Key=Project,Value=voiceguard}]" \
  --user-data "$(base64 -w0 <<'USERDATA'
#!/bin/bash
set -e
# TODO: insert training bootstrap script here
# Must: source checkpoints from S3, train, push checkpoints, then:
aws ec2 terminate-instances --region ap-southeast-1 --instance-ids $(curl -s http://169.254.169.254/latest/meta-data/instance-id)
USERDATA
)" \
  --query 'Instances[0].InstanceId' --output text)

echo "Launched: $INSTANCE_ID"
echo "Monitor: aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID"
```

Show the script in full. Remind the user:
- Fill in AMI_ID, KEY_NAME, SECURITY_GROUP before running
- Script auto-terminates on completion
- Run `chmod +x scripts/aws/launch_<task>.sh` before executing
