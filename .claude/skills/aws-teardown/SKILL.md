---
name: aws-teardown
description: List running EC2 instances and offer to terminate. Requires explicit user confirmation.
disable-model-invocation: true
allowed-tools:
  - Bash(aws ec2 describe-instances *)
---

## Instructions

1. List all running EC2 instances in ap-southeast-1:
   ```bash
   aws ec2 describe-instances \
     --region ap-southeast-1 \
     --filters "Name=instance-state-name,Values=running" \
     --query "Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime,Tags[?Key=='Name'].Value|[0]]" \
     --output table
   ```

2. Show the terminate command for each instance but **do not run it**:
   ```
   aws ec2 terminate-instances --region ap-southeast-1 --instance-ids <id>
   ```

3. Say: "Type 'yes go' to terminate all listed instances, or specify instance IDs."

4. **Only terminate after the user types "yes go"** — not "yes", not approval of a tool call.
