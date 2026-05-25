#!/usr/bin/env bash
# Launch DSFNet training on g5.xlarge Spot in ap-southeast-1.
# DO NOT RUN without reviewing cost estimate below.
#
# Cost estimate: ~$4-5 (8-12 hours @ $0.53/hr Spot for g5.xlarge)
# Instance: g5.xlarge — 1x A10G GPU 24GB, 4 vCPU, 16GB RAM
# AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.1 (Ubuntu 20.04)

set -euo pipefail

REGION="ap-southeast-1"
INSTANCE_TYPE="g5.xlarge"
S3_BUCKET="voiceguard-mt-2026"
KEY_NAME="${KEY_NAME:-voiceguard-training}"  # EC2 key pair name
SPOT_MAX_PRICE="0.60"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  DSFNet Training Launch                              ║"
echo "║  Instance : g5.xlarge Spot                          ║"
echo "║  Region   : ap-southeast-1                          ║"
echo "║  Est. time: 8-12 hours                              ║"
echo "║  Est. cost: ~\$4-5                                   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "This will launch an AWS EC2 instance and charge your account."
echo "Press Ctrl+C within 10 seconds to cancel..."
for i in $(seq 10 -1 1); do printf "\r  Launching in %2d seconds..." $i; sleep 1; done
echo ""

# Fetch latest Deep Learning AMI (PyTorch 2.1, Ubuntu 20.04)
AMI_ID=$(aws ec2 describe-images \
  --region "$REGION" \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.1 (Ubuntu 20.04)*" \
  --query "sort_by(Images,&CreationDate)[-1].ImageId" \
  --output text)

echo "AMI: $AMI_ID"

# User data script — runs on instance startup
USER_DATA=$(cat <<'EOF'
#!/bin/bash
set -ex
exec > /var/log/voiceguard-training.log 2>&1

# Install project
pip install --upgrade pip
pip install \
  "torch>=2.1.0" "torchaudio>=2.1.0" \
  "scikit-learn>=1.4.0" "xgboost>=2.0.3" \
  "numpy>=1.26.0" "scipy>=1.11.0" \
  "boto3>=1.34.0"

# Pull latest source
aws s3 cp s3://voiceguard-mt-2026/src/voiceguard.tar.gz /tmp/voiceguard.tar.gz
mkdir -p /opt/voiceguard
tar -xzf /tmp/voiceguard.tar.gz -C /opt/voiceguard
pip install -e /opt/voiceguard --no-deps

# Pull preprocessed data
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

# Signal completion and self-terminate
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --region ap-southeast-1 --instance-ids "$INSTANCE_ID"
EOF
)

# Launch Spot instance
LAUNCH_SPEC=$(cat <<SPEC
{
  "ImageId": "$AMI_ID",
  "InstanceType": "$INSTANCE_TYPE",
  "KeyName": "$KEY_NAME",
  "IamInstanceProfile": {"Name": "voiceguard-training-role"},
  "UserData": "$(echo "$USER_DATA" | base64 -w0)",
  "BlockDeviceMappings": [{
    "DeviceName": "/dev/sda1",
    "Ebs": {"VolumeSize": 100, "VolumeType": "gp3"}
  }]
}
SPEC
)

SPOT_REQUEST=$(aws ec2 request-spot-instances \
  --region "$REGION" \
  --spot-price "$SPOT_MAX_PRICE" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification "$LAUNCH_SPEC" \
  --query "SpotInstanceRequests[0].SpotInstanceRequestId" \
  --output text)

echo "Spot request: $SPOT_REQUEST"

# CloudWatch alarm: CPU < 5% for 30 min → stop instance
# (Set up after instance ID is known — see monitor_training.sh)

echo "Monitor: aws ec2 describe-spot-instance-requests --region $REGION --spot-instance-request-ids $SPOT_REQUEST"
echo "Logs stream to s3://$S3_BUCKET/checkpoints/dsfnet/training_history.json"
