# DNS Setup

Point your domain at this server's public IP before running `deploy/setup.sh`
(Certbot needs the domain to resolve to this host to issue the certificate).

At your registrar (Namecheap / Cloudflare / GoDaddy) add:

| Type | Host | Value          | TTL  |
|------|------|----------------|------|
| A    | @    | YOUR_SERVER_IP | Auto |
| A    | www  | YOUR_SERVER_IP | Auto |

> If you use Cloudflare, set the records to **DNS only** (grey cloud) until the
> certificate is issued, otherwise Cloudflare's proxy interferes with the
> Certbot HTTP-01 challenge. You can re-enable the proxy afterwards.

Wait 5–30 minutes for DNS to propagate, then confirm it resolves to your IP:

```bash
nslookup voice-deepfake-vishing-detector-generator.eu.cc
dig +short voice-deepfake-vishing-detector-generator.eu.cc
```

## Find your public IP

```bash
curl ifconfig.me
```

## Open required ports

If running on a home PC, port-forward in your router:

```
External 80  -> PC_LOCAL_IP:80
External 443 -> PC_LOCAL_IP:443
```

Find your PC's local IP:

```bash
# Linux
ip addr show | grep "inet "
# Windows
ipconfig
```

Only ports 80 and 443 need to be public. The FastAPI backend stays bound to
`127.0.0.1:8000` and is reached only through Nginx — never expose 8000 directly.
