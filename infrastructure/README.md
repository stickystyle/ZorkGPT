# ZorkGPT Viewer Infrastructure

This directory contains AWS CDK infrastructure code to deploy the ZorkGPT Live Viewer to AWS. The infrastructure includes:

- **S3 Bucket**: Hosts the website files and state JSON files
- **CloudFront Distribution**: Provides global CDN with custom caching rules
- **EC2 Instance**: t2.micro instance that runs ZorkGPT automatically
- **IAM Role**: Permissions for EC2 to upload to S3

## 🚀 Quick Start

### Prerequisites

1. **AWS CLI** installed and configured
   ```bash
   aws configure
   ```

2. **Node.js** (required for AWS CDK)
   - Download from [nodejs.org](https://nodejs.org/)

3. **Python 3.8+** with pip

### One-Command Deployment

```bash
cd infrastructure
python deploy.py
```

This script will:
- Check all prerequisites
- Install AWS CDK if needed
- Set up Python environment
- Bootstrap CDK in your AWS account
- Deploy the infrastructure
- Display all the important URLs and credentials

## 📋 Manual Deployment Steps

If you prefer to run the steps manually:

### 1. Install Dependencies

```bash
cd infrastructure

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install AWS CDK (if not already installed)
npm install -g aws-cdk
```

### 2. Bootstrap CDK

```bash
# Bootstrap CDK in your AWS account (one-time setup)
cdk bootstrap
```

### 3. Deploy the Stack

```bash
# Synthesize the CloudFormation template
cdk synth

# Deploy the infrastructure
cdk deploy
```

### 4. Get Stack Outputs

After deployment, note the outputs:
- **ViewerURL**: Your live viewer website URL  
- **BucketName**: S3 bucket name (automatically configured on EC2)
- **EC2PublicIP**: IP address of your ZorkGPT instance
- **EC2SSHCommand**: Ready-to-use SSH command to connect

## 🖥️ EC2 Instance

The deployment includes a t2.micro EC2 instance that automatically:

- **Clones ZorkGPT** from the GitHub repository
- **Sets up Python environment** with all dependencies
- **Configures S3 integration** automatically
- **Creates a systemd service** for auto-restart
- **Runs continuously** uploading state to S3

### Connecting to Your Instance

Use the SSH command from the stack outputs:
```bash
ssh -i ~/.ssh/parrishfamily.pem ec2-user@YOUR-INSTANCE-IP
```

### Managing ZorkGPT on EC2

#### Remote Management (Recommended)
Use the management script to control your instance remotely:

```bash
cd infrastructure

# Get instance information
python manage_ec2.py info

# Check ZorkGPT status
python manage_ec2.py status

# Start/stop/restart ZorkGPT
python manage_ec2.py start
python manage_ec2.py stop
python manage_ec2.py restart

# View logs
python manage_ec2.py logs
python manage_ec2.py logs-follow  # Follow logs in real-time

# Update ZorkGPT to latest version
python manage_ec2.py update

# SSH to the instance
python manage_ec2.py ssh
```

#### Direct SSH Management
If you prefer to SSH directly:

```bash
# Check if ZorkGPT is running
sudo systemctl status zorkgpt

# Start ZorkGPT service
sudo systemctl start zorkgpt

# Stop ZorkGPT service
sudo systemctl stop zorkgpt

# View logs
sudo journalctl -u zorkgpt -f

# Manual run (as zorkgpt user)
sudo su - zorkgpt
cd ZorkGPT
./start_zorkgpt.sh
```

## 🔧 Local Integration (Optional)

You can also run ZorkGPT locally with S3 integration by setting the environment variable:

```bash
export ZORK_S3_BUCKET=your-bucket-name-from-stack-output
```

The S3 bucket is now configured exclusively through the `ZORK_S3_BUCKET` environment variable to avoid git conflicts during server updates. This means you can run the same codebase both locally and on the server without configuration file differences.

## 🌐 CloudFront Caching

The CloudFront distribution is configured with smart caching:

- **`current_state.json`**: No caching (always fresh)
- **HTML files**: 5-minute cache with cache-busting support
- **Other static assets**: 1-hour cache

This ensures your live data is always up-to-date while static assets load quickly.

## 💰 Cost Considerations

This infrastructure is designed to be cost-effective:

- **S3**: Pay only for storage and requests (minimal for JSON files)
- **CloudFront**: Free tier includes 1TB of data transfer
- **Price Class 100**: Uses only North America and Europe edge locations

Estimated monthly cost for typical usage: **$10-15 USD**
- t2.micro EC2 instance: ~$8.50/month (free tier eligible for first year)
- S3 storage and requests: ~$1-2/month
- CloudFront: Free tier covers most usage

## 🔒 Security Features

- **Origin Access Control**: S3 bucket is not publicly accessible
- **HTTPS Only**: All traffic redirected to HTTPS
- **IAM Role**: EC2 instance uses role-based permissions (no hardcoded keys)
- **Security Group**: SSH access only (port 22)
- **No Public Write**: Only authorized uploads allowed

## 🛠️ Customization

### Change Caching Behavior

Edit `zorkgpt_viewer_stack.py` to modify cache policies:

```python
# Example: Increase HTML cache time
default_ttl=Duration.minutes(10),  # Changed from 5 to 10 minutes
```

### Add Custom Domain

To use your own domain, add this to the CloudFront distribution:

```python
domain_names=["viewer.yourdomain.com"],
certificate=acm.Certificate.from_certificate_arn(
    self, "Certificate", "arn:aws:acm:us-east-1:123456789012:certificate/..."
),
```

### Change Regions

The stack uses your default AWS region. To deploy to a specific region:

```bash
cdk deploy --context region=us-west-2
```

## 🧹 Cleanup

To remove all infrastructure:

```bash
cdk destroy
```

**Warning**: This will delete the S3 bucket and all files. Make sure to backup any important data first.

## 📊 Monitoring

### CloudWatch Metrics

Monitor your deployment through AWS CloudWatch:
- CloudFront request counts and error rates
- S3 storage usage and request metrics

### Access Logs

Enable CloudFront access logs by adding to the distribution:

```python
enable_logging=True,
log_bucket=log_bucket,
log_file_prefix="cloudfront-logs/",
```

## 🐛 Troubleshooting

### Common Issues

1. **CDK Bootstrap Error**
   ```
   Solution: Run `cdk bootstrap` in your target region
   ```

2. **Permission Denied**
   ```
   Solution: Check AWS credentials with `aws sts get-caller-identity`
   ```

3. **Upload Fails**
   ```
   Solution: Verify bucket name and access keys from stack outputs
   ```

4. **Website Not Loading**
   ```
   Solution: Check CloudFront distribution status (takes 5-15 minutes to deploy)
   ```

### Debug Commands

```bash
# Check stack status
aws cloudformation describe-stacks --stack-name ZorkGPTViewerStack

# List S3 bucket contents
aws s3 ls s3://your-bucket-name/

# Check CloudFront distribution
aws cloudfront list-distributions
```

## 📚 Additional Resources

- [AWS CDK Documentation](https://docs.aws.amazon.com/cdk/)
- [CloudFront Caching Behavior](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Expiration.html)
- [S3 Static Website Hosting](https://docs.aws.amazon.com/AmazonS3/latest/userguide/WebsiteHosting.html)

## 🤝 Contributing

To modify the infrastructure:

1. Make changes to `zorkgpt_viewer_stack.py`
2. Test with `cdk synth`
3. Deploy with `cdk deploy`
4. Update this README if needed

## 📄 License

This infrastructure code follows the same license as the main ZorkGPT project.