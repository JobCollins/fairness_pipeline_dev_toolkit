# Security Policy

## Supported Versions

We actively support security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| 0.4.x   | :x:                |
| < 0.4   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue. Instead, please report it via one of the following methods:

1. **Email**: Send details to the maintainers (see repository contacts)
2. **GitHub Security Advisory**: Use GitHub's private vulnerability reporting feature if available
3. **Private Issue**: Create a private security issue if you have repository access

### What to Include

When reporting a vulnerability, please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)
- Any relevant CVE numbers or security advisories

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution**: Depends on severity (see below)

## Security Severity Levels

### Critical
- Remote code execution
- Authentication bypass
- Data exfiltration
- **Response Time**: Immediate (within 24 hours)

### High
- Privilege escalation
- Significant data exposure
- Denial of service
- **Response Time**: Within 7 days

### Medium
- Information disclosure
- Limited denial of service
- **Response Time**: Within 30 days

### Low
- Minor information disclosure
- Best practice violations
- **Response Time**: Next release cycle

## Security Practices

### Dependency Management

We maintain security through:

1. **Regular Security Scans**
   - Automated weekly scans via GitHub Actions
   - Manual scans before releases
   - Using `pip-audit` for vulnerability detection

2. **Dependency Updates**
   - Critical and high-severity vulnerabilities are addressed immediately
   - Medium-priority vulnerabilities are addressed monthly
   - All dependencies are pinned to secure versions in `pyproject.toml`

3. **Transitive Dependencies**
   - We pin security-critical indirect dependencies
   - Regular review of dependency tree
   - Automated alerts for new vulnerabilities

### Current Security Measures

- **Automated Security Scanning**: Weekly scans via `.github/workflows/security.yml`
- **Dependency Pinning**: Security-critical dependencies pinned in `pyproject.toml`
- **Security Documentation**: `SECURITY_SCAN_RESULTS.md` tracks known vulnerabilities
- **CI/CD Integration**: Security checks run on all pull requests

### Security Workflow

1. **Detection**: Automated scans or manual reports
2. **Assessment**: Evaluate severity and impact
3. **Fix**: Update dependencies or patch code
4. **Verification**: Re-run security scans
5. **Documentation**: Update `SECURITY_SCAN_RESULTS.md`
6. **Release**: Deploy fix in appropriate release

## Known Vulnerabilities

See `SECURITY_SCAN_RESULTS.md` for a detailed list of known vulnerabilities and their remediation status.

## Security Updates

Security updates are released as:
- **Patch releases** (0.5.x → 0.5.y) for critical/high severity
- **Minor releases** (0.5.x → 0.6.0) for medium/low severity bundled with other updates

## Best Practices for Users

1. **Keep Dependencies Updated**
   ```bash
   pip install --upgrade fairness-pipeline-dev-toolkit
   ```

2. **Review Security Advisories**
   - Check `SECURITY_SCAN_RESULTS.md` regularly
   - Subscribe to security notifications

3. **Use Virtual Environments**
   - Isolate dependencies
   - Use `virtualenv>=20.36.2` or later

4. **Regular Security Audits**
   ```bash
   pip install pip-audit
   pip-audit --desc
   ```

## Security Contacts

For security-related questions or reports:
- **Repository**: [GitHub Issues](https://github.com/SvrusIO/fAIr/issues)
- **Security Issues**: Use private reporting methods listed above

## Acknowledgments

We thank security researchers and users who responsibly report vulnerabilities. Contributors will be acknowledged (with permission) in security advisories and release notes.

---

**Last Updated**: 2025-01-24  
**Next Review**: 2025-02-24
