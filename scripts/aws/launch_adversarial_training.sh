#!/usr/bin/env bash
# Launch adversarial hardening on g4dn.xlarge Spot instance.
# Estimated cost: ~$0.32/hr × 6 hrs = ~$1.92
# MUST confirm with user before executing.
set -euo pipefail

REGION="ap-southeast-1"
INSTANCE_TYPE="g4dn.xlarge"
S3_BUCKET="voiceguard-mt-2026"
KEY_NAME="${KEY_NAME:-voiceGard}"
SPOT_MAX_PRICE="0.40"
AMI_ID="ami-08e82ab44e1c5f952"
SECURITY_GROUP="sg-06e228a510bf8ee6b"
SUBNET_ID="subnet-07c4fe31a713fe5aa"
IAM_PROFILE="instanceRole"
CHECKPOINT_S3="s3://${S3_BUCKET}/checkpoints/dsfnet/dsfnet_best.pt"
OUTPUT_PREFIX="s3://${S3_BUCKET}/checkpoints/adversarial"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Adversarial Hardening Launch                        ║"
echo "║  Instance : g4dn.xlarge Spot (T4 GPU 16GB)          ║"
echo "║  Region   : ap-southeast-1b                         ║"
echo "║  Est. time: 6 hours                                 ║"
echo "║  Est. cost: ~\$1.92                                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Requires dsfnet_best.pt in S3 (run launch_dsfnet_training.sh first)."
echo "Press Ctrl+C within 10 seconds to cancel..."
for i in $(seq 10 -1 1); do printf "\r  Launching in %2d seconds..." $i; sleep 1; done
echo ""

USER_DATA=$(base64 -w0 <<USERDATA
#!/bin/bash
set -ex
exec > /var/log/voiceguard-adversarial.log 2>&1

pip install --upgrade pip
pip install "scikit-learn>=1.4.0" "xgboost>=2.0.3" "numpy>=1.26.0" \
            "scipy>=1.11.0" "boto3>=1.34.0" "librosa>=0.10.0"

aws s3 cp s3://voiceguard-mt-2026/src/voiceguard.tar.gz /tmp/voiceguard.tar.gz
mkdir -p /opt/voiceguard
tar -xzf /tmp/voiceguard.tar.gz -C /opt/voiceguard
pip install -e /opt/voiceguard --no-deps

# Download base checkpoint
aws s3 cp ${CHECKPOINT_S3} /tmp/dsfnet_best.pt

# Pull data
aws s3 sync s3://voiceguard-mt-2026/data/asvspoof2021/ /data/asvspoof2021/

# Run adversarial training
cd /opt/voiceguard
python -c "
import torch
from voiceguard.models.dsfnet import DSFNet
from voiceguard.training.adversarial import adversarial_training_step
from voiceguard.training.train_dsfnet import AudioDataset
import subprocess, os

model = DSFNet()
ckpt = torch.load('/tmp/dsfnet_best.pt', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.cuda()

dataset = AudioDataset('/data/asvspoof2021/tensors')
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

for epoch in range(10):
    losses = []
    for wav, spec, labels in loader:
        wav, labels = wav.cuda(), labels.cuda()
        loss = adversarial_training_step(model, optimizer, wav, labels, epsilon=0.01, pgd_steps=5)
        losses.append(loss)
    print(f'Epoch {epoch+1}/10  loss={sum(losses)/len(losses):.4f}')
    ckpt_path = f'/tmp/adv_epoch_{epoch+1}.pt'
    torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'loss': sum(losses)/len(losses)}, ckpt_path)
    subprocess.run(['aws', 's3', 'cp', ckpt_path, '${OUTPUT_PREFIX}/'], check=False)

print('Adversarial training complete')
subprocess.run(['aws', 's3', 'cp', '/tmp/adv_epoch_10.pt', '${OUTPUT_PREFIX}/dsfnet_adversarial_best.pt'], check=False)
"

INSTANCE_ID=\$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --region ${REGION} --instance-ids "\$INSTANCE_ID"
USERDATA
)

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
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Project,Value=VoiceGuard},{Key=Phase,Value=adversarial-training}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo ""
echo "✓ Instance launched: $INSTANCE_ID"
echo ""
echo "Monitor commands:"
echo "  State : aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text"
echo "  Output: aws s3 ls ${OUTPUT_PREFIX}/"
