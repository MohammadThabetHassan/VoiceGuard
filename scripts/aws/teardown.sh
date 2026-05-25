#!/usr/bin/env bash
# List running EC2 instances and interactively terminate them.
# Usage: bash scripts/aws/teardown.sh

set -uo pipefail

REGION="ap-southeast-1"

echo ""
echo "═══════════════════════════════════════"
echo "  VoiceGuard AWS Teardown"
echo "  Region: $REGION"
echo "═══════════════════════════════════════"
echo ""

if ! command -v aws &>/dev/null; then
  echo "Error: aws CLI not installed"
  exit 1
fi

if ! aws sts get-caller-identity &>/dev/null; then
  echo "Error: AWS not authenticated. Attach IAM role or run 'aws configure'"
  exit 1
fi

echo "Running instances:"
INSTANCES=$(aws ec2 describe-instances \
  --region "$REGION" \
  --filters "Name=instance-state-name,Values=running" \
  --query "Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime,Tags[?Key=='Name'].Value|[0]]" \
  --output table 2>&1)

echo "$INSTANCES"

IDS=$(aws ec2 describe-instances \
  --region "$REGION" \
  --filters "Name=instance-state-name,Values=running" \
  --query "Reservations[].Instances[].InstanceId" \
  --output text 2>/dev/null)

if [[ -z "$IDS" ]]; then
  echo "No running instances found."
  exit 0
fi

echo ""
echo "Terminate command (NOT running it yet):"
echo "  aws ec2 terminate-instances --region $REGION --instance-ids $IDS"
echo ""
read -r -p "Type 'yes go' to terminate all listed instances: " CONFIRM

if [[ "$CONFIRM" == "yes go" ]]; then
  aws ec2 terminate-instances --region "$REGION" --instance-ids $IDS
  echo "Termination requested for: $IDS"
else
  echo "Aborted. No instances terminated."
fi
