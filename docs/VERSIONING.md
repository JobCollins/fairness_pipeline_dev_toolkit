# Versioning Strategy and Backward Compatibility Policy

**Last Updated:** 2026-05-14  
**Current Version:** 0.9.0

---

## Overview

The Fairness Pipeline Development Toolkit follows [Semantic Versioning (SemVer)](https://semver.org/spec/v2.0.0.html) to communicate the nature of changes in each release. This document outlines our versioning strategy, backward compatibility guarantees, deprecation policy, and migration guidance.

---

## Semantic Versioning

Version numbers follow the format: **MAJOR.MINOR.PATCH** (e.g., `0.5.0`)

### Version Number Components

- **MAJOR** (X.0.0): Breaking changes that require user code modifications
- **MINOR** (0.X.0): New features, enhancements, or additions that remain backward compatible
- **PATCH** (0.0.X): Bug fixes, security patches, or minor improvements that don't change behavior

### Current Status

- **Current Version:** `0.8.0` (Beta)
- **Development Status:** Beta (pre-1.0.0)
- **Pre-1.0.0 Policy:** During the 0.x phase, MINOR version increments may include breaking changes. Once we reach 1.0.0, strict SemVer will be enforced.

---

## Backward Compatibility Guarantees

### Public API Stability

**What is considered "Public API"?**

The following are considered public APIs and maintain backward compatibility within the same major version:

1. **Public Classes and Functions** exported in `__init__.py` files:
   - `fairpipe.FairnessAnalyzer`
   - `fairpipe.MetricResult`
   - `fairpipe.assert_fairness`
   - `fairpipe.log_fairness_metrics`
   - `fairpipe.to_markdown_report`
   - All classes and functions listed in `__all__` of public modules

2. **CLI Commands and Arguments:**
   - `fairpipe version`
   - `fairpipe validate` (with existing arguments)
   - `fairpipe pipeline` (with existing arguments)
   - `fairpipe run-pipeline` (with existing arguments)
   - `fairpipe train-regularized`, `fairpipe train-lagrangian`, `fairpipe calibrate`

3. **Configuration File Schema:**
   - YAML configuration structure for `PipelineConfig`
   - Field names and types in configuration files
   - Default values and behavior

4. **Data Formats:**
   - Input/output CSV formats
   - JSON artifact formats (reports, metrics)
   - MLflow logging structure

### Compatibility Guarantees by Version Type

#### MAJOR Version (X.0.0)
- **Breaking Changes:** Public APIs may change in incompatible ways
- **Migration Required:** User code modifications necessary
- **Deprecation Period:** Breaking changes are announced at least one MINOR version in advance
- **Documentation:** Comprehensive migration guide provided

#### MINOR Version (0.X.0)
- **Backward Compatible:** Existing code continues to work without modification
- **New Features:** New functionality added without breaking existing APIs
- **Additive Changes:** New optional parameters, new classes, new methods
- **Configuration:** New optional configuration fields may be added

#### PATCH Version (0.0.X)
- **Bug Fixes Only:** No API changes, no new features
- **Behavior Corrections:** Fixes to incorrect behavior
- **Security Patches:** Security vulnerabilities addressed
- **Internal Improvements:** Performance optimizations, code refactoring (no external impact)

---

## Deprecation Policy

### Deprecation Process

1. **Announcement:** Deprecated features are announced in the CHANGELOG with a deprecation notice
2. **Warning Period:** Deprecated features emit warnings when used (via `warnings` module)
3. **Documentation:** Deprecation notices include:
   - Reason for deprecation
   - Recommended alternative
   - Timeline for removal
   - Migration examples

4. **Removal Timeline:**
   - **Pre-1.0.0:** Deprecated features may be removed in the next MINOR version
   - **Post-1.0.0:** Deprecated features will be removed in the next MAJOR version (minimum 2 MINOR versions notice)

### Example Deprecation Notice

```python
import warnings

def old_function():
    warnings.warn(
        "old_function() is deprecated and will be removed in v1.0.0. "
        "Use new_function() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... implementation
```

---

## Public API Boundaries

### Stable Public APIs

These modules and their public exports are stable:

#### Core Metrics (`fairpipe.metrics`)
- `FairnessAnalyzer` class and all its public methods
- `MetricResult` class
- Metric computation methods (signatures and return types)

#### Pipeline (`fairpipe.pipeline`)
- `PipelineConfig` class
- `load_config()` function
- `build_pipeline()`, `apply_pipeline()`, `run_detectors()` functions
- Transformer classes: `InstanceReweighting`, `DisparateImpactRemover`, `ReweighingTransformer`, `ProxyDropper`

#### Integration (`fairpipe.integration`)
- `execute_workflow()` function
- `WorkflowResult`, `ValidationResult` classes
- `log_workflow_results()` function
- `to_markdown_report()` function
- `assert_fairness()` pytest plugin

#### Training (`fairpipe.training`)
- Training wrapper classes and their public methods
- Calibration functions

#### Monitoring (`fairpipe.monitoring`)
- `RealTimeFairnessTracker` class
- `FairnessDriftAndAlertEngine` class
- `FairnessReportingDashboard` class

#### Exceptions (`fairpipe.exceptions`)
- Exception class hierarchy

### Internal APIs (Not Guaranteed)

The following are **internal** and may change without notice:

- Modules not listed in public `__init__.py` files
- Private methods (prefixed with `_`)
- Internal utility functions
- Implementation details in submodules (e.g., `orchestration.registry`, `detectors.core`)

**Recommendation:** Import only from public modules listed in `__init__.py` files.

---

## Configuration File Compatibility

### Configuration Schema Evolution

- **MAJOR Changes:** Schema restructuring, required field changes, incompatible field renames
- **MINOR Changes:** New optional fields added, default values changed (backward compatible)
- **PATCH Changes:** Bug fixes in config parsing, validation improvements

### Migration Strategy

- **Backward Compatibility:** Old configuration files continue to work unless explicitly stated
- **Validation:** Invalid configurations produce clear error messages
- **Examples:** Updated configuration examples provided in documentation

### Example: Configuration Evolution

**v0.5.0** (previous):
```yaml
pipeline:
  transformers:
    - type: InstanceReweighting
      sensitive: ["gender"]
```

**v0.6.0** (hypothetical - adds optional field):
```yaml
pipeline:
  transformers:
    - type: InstanceReweighting
      sensitive: ["gender"]
      # New optional field
      weight_threshold: 0.1
```

The v0.5.0 config remains valid in v0.6.0.

---

## CLI Compatibility

### Command Stability

- **Command Names:** Stable within major version
- **Command Arguments:** Existing arguments remain supported
- **New Arguments:** Added as optional in MINOR versions
- **Output Format:** JSON/CSV output formats remain stable

### Breaking Changes

If CLI changes are necessary:
1. Deprecation warning issued in previous version
2. Old command/argument remains functional for one MINOR version
3. Migration guide provided
4. Breaking change occurs in next MAJOR version

---

## Data Format Compatibility

### Input Formats

- **CSV Input:** Column names and data types remain stable
- **JSON Artifacts:** Structure remains backward compatible (new fields may be added)

### Output Formats

- **Report Formats:** Markdown report structure remains stable
- **Metrics JSON:** Core fields remain stable, new fields may be added
- **MLflow Logging:** Logging structure remains compatible

---

## Python Version Support

### Supported Python Versions

- **Current:** Python 3.10, 3.11, 3.12
- **Policy:** Support for Python versions follows [Python's release cycle](https://devguide.python.org/versions/)
- **Dropping Support:** Dropped Python versions are announced in CHANGELOG at least one MINOR version in advance

### Dependency Compatibility

- **Minimum Versions:** Specified in `pyproject.toml` (e.g., `numpy>=1.26`)
- **Maximum Versions:** Not typically constrained unless breaking changes occur
- **Optional Dependencies:** May change without affecting core functionality

---

## Migration Guides

### When Migrating Between Versions

1. **Check CHANGELOG:** Review `CHANGELOG.md` for breaking changes
2. **Review Migration Notes:** Look for "Migration Notes" sections in release notes
3. **Test Incrementally:** Test your code with the new version
4. **Update Dependencies:** Ensure compatible dependency versions

### Example Migration Scenarios

#### Scenario 1: MINOR Version Update (0.5.0 → 0.6.0)

**Action:** Update version, test existing code
- No code changes required
- New features available but not required
- Existing functionality unchanged

#### Scenario 2: MAJOR Version Update (0.5.0 → 1.0.0)

**Action:** Review breaking changes, update code
- Check CHANGELOG for breaking changes
- Review migration guide
- Update deprecated API calls
- Test thoroughly

---

## Version Information

### Checking Version

**Programmatically:**
```python
from fairpipe import __version__
print(__version__)  # "0.6.2"
```

**CLI:**
```bash
fairpipe version
# Output: 0.6.2
```

**Package Metadata:**
```bash
pip show fairpipe
```

---

## Release Process

### Version Bumping

1. **PATCH:** Bug fixes, security patches → increment PATCH
2. **MINOR:** New features, enhancements → increment MINOR, reset PATCH
3. **MAJOR:** Breaking changes → increment MAJOR, reset MINOR and PATCH

### Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update `__version__` in `__init__.py`
- [ ] Update `CHANGELOG.md` with release notes
- [ ] Tag release in git: `git tag v0.5.0`
- [ ] Build and publish to PyPI
- [ ] Update documentation if needed

---

## Pre-1.0.0 Considerations

### Current Status: Beta (0.6.2)

During the 0.x phase:

- **Flexibility:** MINOR versions may include breaking changes (with notice)
- **Rapid Iteration:** API evolution based on user feedback
- **Stability Goal:** Move toward 1.0.0 with stable APIs

### Path to 1.0.0

1. **API Stabilization:** Finalize public API surface
2. **Documentation:** Complete API documentation
3. **Testing:** Comprehensive test coverage
4. **User Feedback:** Incorporate feedback from beta users
5. **Breaking Changes:** Address any remaining breaking changes before 1.0.0

### Post-1.0.0

Once version 1.0.0 is released:
- **Strict SemVer:** Full backward compatibility guarantees
- **Long-term Support:** Extended support for major versions
- **Stable APIs:** Public APIs remain stable within major versions

---

## Best Practices for Users

### Version Pinning

**For Production:**
```txt
fairpipe==0.6.2
```

**For Development:**
```txt
fairpipe>=0.6.2,<1.0.0
```

### Staying Updated

1. **Subscribe to Releases:** Watch GitHub repository for release notifications
2. **Review CHANGELOG:** Check `CHANGELOG.md` before updating
3. **Test in Staging:** Test new versions in non-production environments first
4. **Report Issues:** Report compatibility issues via GitHub Issues

### Avoiding Breaking Changes

- **Use Public APIs:** Import from public modules only
- **Avoid Private APIs:** Don't import from internal modules
- **Follow Documentation:** Use documented APIs and patterns
- **Test Regularly:** Keep dependencies updated and test frequently

---

## Support and Questions

### Reporting Compatibility Issues

If you encounter backward compatibility issues:

1. **Check Version:** Verify you're using a supported version
2. **Review CHANGELOG:** Check if the issue is a known breaking change
3. **File Issue:** Report on [GitHub Issues](https://github.com/SvrusIO/fAIr/issues)
4. **Include Details:** Version numbers, error messages, code examples

### Getting Help

- **Documentation:** See [API Reference](api.md) and [Integration Guide](integration_guide.md)
- **GitHub:** [Repository](https://github.com/SvrusIO/fAIr)
- **Issues:** [GitHub Issues](https://github.com/SvrusIO/fAIr/issues)

---

## Summary

- **Versioning:** Semantic Versioning (SemVer)
- **Current Version:** 0.6.2 (Beta)
- **Backward Compatibility:** Guaranteed within major versions (post-1.0.0)
- **Public APIs:** Stable within major versions
- **Deprecation:** Minimum notice period before removal
- **Migration:** Guides provided for breaking changes

This policy ensures predictable, stable releases while allowing the toolkit to evolve and improve based on user needs and feedback.
