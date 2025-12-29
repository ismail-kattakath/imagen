# Security Implementation Guide

This document describes the security measures implemented in the Imagen platform.

## Overview

The Imagen platform implements defense-in-depth security with multiple layers:

1. **Network Security** - Private VPC, Network Policies, Cloud Armor WAF
2. **Secrets Management** - Google Cloud Secret Manager
3. **Access Control** - Resource-level IAM, Workload Identity
4. **Data Protection** - Encryption at rest, CORS restrictions
5. **Monitoring** - Audit logs, Security Command Center integration

---

## 1. Network Security

### Private VPC Architecture

**Implementation**: `terraform/modules/vpc/`

All GKE nodes run in a private VPC with no public IP addresses:
- **GKE Subnet**: `10.0.0.0/20` (private nodes)
- **Pod Range**: `10.4.0.0/14` (secondary range)
- **Service Range**: `10.8.0.0/20` (secondary range)
- **Cloud NAT**: Provides outbound internet access
- **VPC Flow Logs**: Enabled for traffic monitoring

**Benefits**:
- Nodes not directly accessible from internet
- All egress traffic goes through Cloud NAT
- Complete network traffic visibility

### Kubernetes Network Policies

**Implementation**: `k8s/network-policies/`

Default-deny network policies with explicit allow rules:

```yaml
# Default: Deny all ingress/egress
- default-deny-all

# Allow: DNS resolution (required)
- allow-dns

# Allow: Worker → Triton communication
- allow-triton-ingress
- allow-worker-to-triton
```

**Traffic Flow**:
1. Workers can only communicate with Triton on ports 8000/8001
2. All pods can query DNS (kube-dns)
3. Workers can access GCP APIs (Pub/Sub, GCS) on port 443
4. All other traffic is blocked by default

### Cloud Armor WAF

**Implementation**: `terraform/modules/cloud-armor/`

Web Application Firewall protecting the Cloud Run API:

**Protection Rules**:
- **Rate Limiting**: 100 requests/minute per IP (configurable)
- **SQL Injection**: Blocks common SQLi patterns
- **XSS**: Blocks cross-site scripting attempts
- **Adaptive Protection**: Auto-blocks DDoS attacks

**Configuration**:
```hcl
module "cloud_armor" {
  source                      = "./modules/cloud-armor"
  project_id                  = var.project_id
  rate_limit_threshold        = 100
  enable_adaptive_protection  = true
}
```

---

## 2. Secrets Management

### Google Cloud Secret Manager

**Implementation**: `terraform/modules/secret-manager/` and `src/utils/secrets.py`

All sensitive data is stored in Secret Manager, not environment variables:

**Secrets**:
- `jwt-secret`: JWT signing key for authentication
- `api-keys`: API keys configuration (JSON)
- `cors-origins`: Allowed CORS origins (comma-separated)

**Access Pattern**:
```python
from src.utils.secrets import get_secret_or_env

# Automatically uses Secret Manager in production, env vars in dev
jwt_secret = get_secret_or_env(project_id, "jwt-secret", "JWT_SECRET")
```

**IAM Access**:
- Cloud Run service account: `secretAccessor` role
- GKE workload identity: `secretAccessor` role for workers
- Secrets are accessed at runtime, never stored in code/config

**Local Development**:
Set `USE_LOCAL_SECRETS=true` to use environment variables instead of Secret Manager.

---

## 3. Access Control

### Private GKE Cluster

**Configuration**: `terraform/modules/gke/main.tf`

```hcl
private_cluster_config {
  enable_private_nodes    = true   # Nodes have no public IPs
  enable_private_endpoint = false  # Keep kubectl access via public endpoint
  master_ipv4_cidr_block  = "172.16.0.0/28"
}
```

**Security Features**:
- Nodes are not publicly accessible
- Master endpoint can be restricted to specific IPs
- Workload Identity enabled (no service account keys)

### VPC Connector for Cloud Run

Cloud Run API connects to private VPC to access GKE services:

```hcl
vpc_access {
  connector = var.vpc_connector_name
  egress    = "PRIVATE_RANGES_ONLY"  # Only private traffic uses VPC
}
```

**Benefits**:
- Cloud Run can communicate with private GKE services
- Public traffic still goes directly to internet
- No exposure of internal services

---

## 4. CORS Security

### Production CORS Configuration

**Validation**: `src/api/main.py`

```python
if settings.is_production() and not settings.cors_origins:
    raise ValueError("CORS_ORIGINS must be set in production")
```

**Requirements**:
- Production MUST set explicit CORS origins (no wildcards)
- Multiple origins supported (comma-separated)
- Credentials allowed for authenticated requests

**Example Configuration**:
```env
# Development
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Production
CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

---

## 5. Additional Security Measures

### Firewall Rules

**Implementation**: `terraform/modules/vpc/main.tf`

- ✅ Allow internal VPC traffic
- ✅ Allow IAP SSH (for debugging)
- ✅ Allow health checks from Google LBs
- ❌ Deny all other ingress by default

### Future Enhancements (LOW Priority)

**Customer-Managed Encryption Keys (CMEK)**:
- Currently using Google-managed encryption
- Can be upgraded to CMEK via `terraform/modules/kms/` (to be implemented)

**Binary Authorization**:
- Container image signing and verification
- Ensures only trusted images run in production
- To be implemented via `terraform/modules/binary-authorization/`

**Enhanced Audit Logging**:
- Currently: Admin Activity logs only
- Future: Data Access logs for all API calls
- Log sink to BigQuery for analysis

---

## Security Checklist

### ✅ Implemented

- [x] Private VPC with Cloud NAT
- [x] Private GKE cluster (no public node IPs)
- [x] VPC connector for Cloud Run
- [x] Kubernetes network policies (default deny)
- [x] Google Cloud Secret Manager integration
- [x] Cloud Armor WAF with rate limiting
- [x] SQL injection & XSS protection
- [x] CORS restrictions (production validation)
- [x] VPC Flow Logs enabled
- [x] Workload Identity for GKE

### ⏳ Planned (Future)

- [ ] Customer-managed encryption keys (CMEK)
- [ ] Binary Authorization for containers
- [ ] Enhanced audit logging (Data Access logs)
- [ ] Security Command Center integration
- [ ] Vulnerability scanning for containers
- [ ] Secret rotation automation

---

## Best Practices

### For Developers

1. **Never commit secrets** - Use Secret Manager or env vars
2. **Test with network policies** - Ensure your service can communicate
3. **Use HTTPS everywhere** - No unencrypted communication
4. **Validate inputs** - All API inputs are validated
5. **Follow least privilege** - Request minimal IAM permissions

### For Operations

1. **Rotate secrets regularly** - JWT secrets, API keys
2. **Review IAM bindings** - Remove unnecessary permissions
3. **Monitor Cloud Armor logs** - Watch for attack patterns
4. **Keep GKE updated** - Use REGULAR release channel
5. **Review network policies** - Adjust as services change

### For Security Audits

1. **Check Secret Manager access logs** - Who accessed what secrets
2. **Review VPC Flow Logs** - Unusual traffic patterns
3. **Analyze Cloud Armor blocks** - Blocked requests and reasons
4. **Audit IAM permissions** - Overly broad access
5. **Verify network policies** - Default deny still in place

---

## Incident Response

### If Secrets Are Compromised

1. Immediately rotate in Secret Manager
2. Check Secret Manager audit logs for unauthorized access
3. Review API logs for unusual activity
4. Update all deployments to use new secrets

### If Attack Detected

1. Check Cloud Armor logs for attack patterns
2. Temporarily lower rate limits if needed
3. Add IP ranges to blocklist via Cloud Armor
4. Review application logs for suspicious requests

### If Unauthorized Access

1. Check IAM audit logs
2. Revoke compromised service account keys
3. Enable Workload Identity if using keys
4. Review all IAM bindings for overly broad access

---

## References

- [GCP VPC Documentation](https://cloud.google.com/vpc/docs)
- [GKE Private Clusters](https://cloud.google.com/kubernetes-engine/docs/how-to/private-clusters)
- [Secret Manager Best Practices](https://cloud.google.com/secret-manager/docs/best-practices)
- [Cloud Armor Overview](https://cloud.google.com/armor/docs/cloud-armor-overview)
- [Kubernetes Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)

---

**Last Updated**: 2025-12-29
**Status**: Production-ready security implementation
