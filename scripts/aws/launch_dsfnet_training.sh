#!/usr/bin/env bash
# Launch DSFNet training on g4dn.xlarge Spot in ap-southeast-1.
# DO NOT RUN without reviewing cost estimate below.
#
# Cost estimate: ~$2.5-3.5 (8-12 hours @ ~$0.32/hr Spot for g4dn.xlarge)
# Instance: g4dn.xlarge — 1x NVIDIA T4 GPU 16GB, 4 vCPU, 16GB RAM
# AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.10 (Amazon Linux 2023)

set -euo pipefail

REGION="ap-southeast-1"
INSTANCE_TYPE="g4dn.xlarge"
S3_BUCKET="voiceguard-mt-2026"
KEY_NAME="${KEY_NAME:-voiceGard}"
SPOT_MAX_PRICE="0.40"
AMI_ID="ami-08e82ab44e1c5f952"
SECURITY_GROUP="sg-06e228a510bf8ee6b"
SUBNET_ID="subnet-07c4fe31a713fe5aa"   # ap-southeast-1b (cheapest AZ)
IAM_PROFILE="instanceRole"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  DSFNet Training Launch                              ║"
echo "║  Instance : g4dn.xlarge Spot (T4 GPU 16GB)          ║"
echo "║  Region   : ap-southeast-1b                         ║"
echo "║  Est. time: 8-12 hours                              ║"
echo "║  Est. cost: ~\$2.50-3.50                             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "This will launch an AWS EC2 instance and charge your account."
echo "Press Ctrl+C within 10 seconds to cancel..."
for i in $(seq 10 -1 1); do printf "\r  Launching in %2d seconds..." $i; sleep 1; done
echo ""

# User data script — runs on instance startup
USER_DATA=$(base64 -w0 <<'USERDATA'
#!/bin/bash
set -ex
exec > /var/log/voiceguard-training.log 2>&1

# Install project dependencies (PyTorch already on DL AMI)
pip install --upgrade pip
pip install \
  "scikit-learn>=1.4.0" "xgboost>=2.0.3" \
  "numpy>=1.26.0" "scipy>=1.11.0" \
  "boto3>=1.34.0" "librosa>=0.10.0" \
  "pytest" "pytest-asyncio"

# Pull latest source from S3
aws s3 cp s3://voiceguard-mt-2026/src/voiceguard.tar.gz /tmp/voiceguard.tar.gz
mkdir -p /opt/voiceguard
tar -xzf /tmp/voiceguard.tar.gz -C /opt/voiceguard
pip install -e /opt/voiceguard --no-deps

# Pull preprocessed ASVspoof tensors
aws s3 sync s3://voiceguard-mt-2026/data/asvspoof2021/ /data/asvspoof2021/

# Run training
cd /opt/voiceguard
python -m voiceguard.training.train_dsfnet \
  --data-path /data/asvspoof2021/tensors \
  --s3-path s3://voiceguard-mt-2026/checkpoints/dsfnet \
  --epochs 40 \
  --batch-size 32 \
  --lr 1e-4 \
  --checkpoint-dir /tmp/checkpoints/dsfnet

# Self-terminate on completion
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --region ap-southeast-1 --instance-ids "$INSTANCE_ID"
USERDATA
)

# Launch Spot instance
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SECURITY_GROUP" \
  --subnet-id "$SUBNET_ID" \
  --iam-instance-profile Name="$IAM_PROFILE" \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$SPOT_MAX_PRICE"'","SpotInstanceType":"one-time"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data "$USER_DATA" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Project,Value=VoiceGuard},{Key=Phase,Value=dsfnet-training}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo ""
echo "✓ Instance launched: $INSTANCE_ID"
echo ""
echo "Monitor commands:"
echo "  State : aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text"
echo "  Log   : aws s3 cp s3://$S3_BUCKET/checkpoints/dsfnet/training_history.json -"
echo "  SSH   : aws ec2 get-console-output --region $REGION --instance-id $INSTANCE_ID --output text"
