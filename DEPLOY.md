# Deploying SentryShield on AWS (EC2)

One Node process serves the pages *and* the WebSocket hub, and it keeps session
state in memory. So the deployment is a single EC2 instance — not a static
site plus a backend, and not an autoscaled fleet.

```
phone / laptop ──HTTPS──▶ CloudFront ──HTTP :8000──▶ EC2 (Node) ──▶ Polly · Bedrock · S3
                └──HTTPS :8443 (self-signed, direct) ─────────────▶
```

Every AWS feature is optional. With no credentials the app runs exactly as it
does locally (browser speech, no Bedrock paragraph, recordings on local disk).

## What AWS is used for

| Service | Used for | Turned on by |
|---|---|---|
| EC2 | Runs the server | — |
| CloudFront | Valid HTTPS on port 443 for phones, no certificate warning | optional |
| Polly | Spoken guidance (`VOICE: AWS POLLY`) | region + credentials |
| Bedrock | One post-scan paragraph, never in the real-time loop | `ENABLE_BEDROCK=true` + `BEDROCK_MODEL_ID` |
| S3 | Durable copy of each finished recording | `S3_BUCKET` |
| IAM | Instance role (no keys on disk) | — |

## 1. IAM role

IAM → Roles → Create role → AWS service → EC2, no managed policies. Then add an
inline policy (JSON tab):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["polly:SynthesizeSpeech", "bedrock:InvokeModel"],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::YOUR-BUCKET/recordings/*"
    }
  ]
}
```

Drop the S3 statement if you are not using the archive.

## 2. S3 bucket (optional)

S3 → Create bucket in the same region, keep **Block all public access** on.
Recordings are uploaded as `recordings/<scan-id>.json`.

## 3. Launch the instance

- Ubuntu Server 24.04 LTS, `t3.small`.
- Security group inbound: TCP **22**, **8000**, **8443**.
- Advanced → IAM instance profile: the role from step 1.
- Attach an **Elastic IP**, so the address (and a CloudFront origin pointing at
  it) survives a stop/start.

Connect with EC2 Instance Connect from the console.

## 4. Install

The shared `.mjs` modules are loaded with `require`, which needs **Node 22+**.

```bash
sudo apt update && sudo apt install -y git
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
git clone https://github.com/prorm/first-commit-.git
cd first-commit-
npm ci
```

## 5. Configure

`.env` is gitignored, so create it on the server. The server reads it at start.

```bash
cp .env.example .env
nano .env
```

With the instance role (recommended — no keys anywhere):

```
AWS_REGION=us-east-1
AWS_USE_INSTANCE_ROLE=true
POLLY_VOICE_ID=Matthew
POLLY_ENGINE=neural
ENABLE_BEDROCK=true
BEDROCK_MODEL_ID=<text model ID or inference profile ID from the Bedrock console>
S3_BUCKET=<your bucket>
```

Notes:

- `AWS_USE_INSTANCE_ROLE=true` is required: a role leaves no environment
  variable, so the adapter cannot detect it on its own.
- One `AWS_REGION` is used for Polly, Bedrock and S3. Pick a region where your
  Bedrock model is available.
- `BEDROCK_MODEL_ID` must be a **text-generation** model (Claude or Titan
  text). Embedding models cannot write the summary.
- Never commit a real `.env`, and never put keys in `.env.example`.
- With static keys instead of a role, set `AWS_ACCESS_KEY_ID` and
  `AWS_SECRET_ACCESS_KEY` and leave `AWS_USE_INSTANCE_ROLE` false.

## 6. Run it

```bash
sudo npm install -g pm2
pm2 start server/index.js --name sentry
pm2 save
pm2 startup        # copy and run the sudo line it prints
```

After any `.env` change: `pm2 restart sentry --update-env`.
To deploy new code: `git pull && npm ci && pm2 restart sentry --update-env`.

## 7. Check it

```bash
curl -s localhost:8000/api/aws
```

Look for `credentialsDetected: true`, `voice.label: "AWS POLLY"`,
`bedrock.enabled: true` and `s3.enabled: true`. Each feature that is off states
its reason. Then finish a scan and confirm `recordings/<scan-id>.json` appears in
the bucket.

| URL | Use |
|---|---|
| `http://<public-ip>:8000/map` | command center on a laptop |
| `https://<public-ip>:8443/phone` | phone (self-signed: Advanced → Proceed) |
| `https://<public-ip>:8443/diagnostics` | capability checks |

The QR code in the server banner shows a private address; ignore it on EC2.

## 8. Valid HTTPS with CloudFront (optional)

CloudFront → Create distribution:

- Origin domain: the instance's **Public IPv4 DNS** (`ec2-….amazonaws.com`).
- Protocol **HTTP only**, HTTP port **8000**.
- Cache policy **CachingDisabled**; origin request policy
  **AllViewerExceptHostHeader**; no WAF.

Use `https://<id>.cloudfront.net/map` and `/phone`. The WebSocket URL is derived
from the page address, so it works through CloudFront unchanged.

## Cost and safety

- A `t3.small` plus a public IPv4 address is roughly $0.60–0.80 per day. Set a
  Billing budget alert.
- There is no login. Anyone with the URL can trigger Polly and Bedrock calls, so
  stop or terminate the instance and release the Elastic IP when the demo is
  over.
- This is a research prototype; see [STATUS.md](STATUS.md) for what it does and
  does not claim.
