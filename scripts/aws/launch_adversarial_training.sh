#!/usr/bin/env bash
# Launch adversarial training on a g5.xlarge Spot instance.
# Estimated cost: ~$1.20/hr × 6 hrs = ~$7.20
# MUST confirm with user before executing.
set -euo pipefail

REGION="ap-southeast-1"
INSTANCE_TYPE="g5.xlarge"
S3_BUCKET="voiceguard-mt-2026"
CHECKPOINT_S3="s3://${S3_BUCKET}/checkpoints/dsfnet_best.pt"
OUTPUT_PREFIX="s3://${S3_BUCKET}/checkpoints/adversarial"

# Fetch latest Deep Learning AMI
AMI_ID=$(aws ec2 describe-images \
  --region "${REGION}" \
  --owners amazon \
  --filters \
    "Name=name,Values=Deep Learning AMI GPU PyTorch*" \
    "Name=state,Values=available" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" \
  --output text)

echo "AMI: ${AMI_ID}"
echo "Instance: ${INSTANCE_TYPE} (Spot)"
echo "Estimated cost: ~\$7.20 for 6 hrs"
echo ""
echo "⚠️  BANNER: This will launch a billable AWS Spot instance."
echo "   Press Ctrl-C within 10 seconds to cancel."
sleep 10

USER_DATA=$(cat <<USERDATA
#!/bin/bash
pip install voiceguard-mt 2>/dev/null || true
python - <<'PYEOF'
import subprocess, sys, boto3

subprocess.run([
    sys.executable, "-m", "voiceguard.training.adversarial_train",
    "--checkpoint", "${CHECKPOINT_S3}",
    "--output_prefix", "${OUTPUT_PREFIX}",
    "--epsilon", "0.01",
    "--pgd_steps", "5",
    "--epochs", "10",
], check=True)
PYEOF

# Self-terminate
INSTANCE_ID=\$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --region ${REGION} --instance-ids "\$INSTANCE_ID"
USERDATA
)

aws ec2 run-instances \
  --region "${REGION}" \
  --image-id "${AMI_ID}" \
  --instance-type "${INSTANCE_TYPE}" \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
  --user-data "${USER_DATA}" \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Project,Value=VoiceGuard},{Key=Phase,Value=adversarial}]" \
  --output json | python3 -c "
import json,sys
d=json.load(sys.stdin)
iid=d['Instances'][0]['InstanceId']
print(f'Launched: {iid}')
"
