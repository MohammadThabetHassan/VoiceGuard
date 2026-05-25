---
name: aws-status
description: Show running AWS resources and month-to-date spend. Use at session start/end to catch forgotten GPU instances.
allowed-tools:
  - Bash(aws *)
---

## Dynamic Context

!`aws ec2 describe-instances --region ap-southeast-1 --filters "Name=instance-state-name,Values=running" --query "Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime,Tags[?Key=='Name'].Value|[0]]" --output table 2>&1`
!`aws ce get-cost-and-usage --time-period Start=$(date -d "$(date +%Y-%m-01)" +%Y-%m-%d),End=$(date +%Y-%m-%d) --granularity MONTHLY --metrics UnblendedCost --query "ResultsByTime[].Total.UnblendedCost.[Amount,Unit]" --output text 2>&1 || echo "Cost Explorer not yet enabled — check AWS Console manually"`

## Instructions

1. List all running EC2 instances in ap-southeast-1.
2. **Alert if any GPU instance (g5.*, p3.*, p4.*) is running** — user may have forgotten to terminate.
3. Report month-to-date spend. Alert thresholds:
   - >$50: "Budget warning"
   - >$100: "Budget alert — >50% spent"
   - >$150: "Budget critical — stop all GPU work"
4. If AWS credentials are not configured, say: "AWS not configured — run `aws configure` or attach IAM role."
