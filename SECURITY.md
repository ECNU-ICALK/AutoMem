# Security Policy

## Supported Versions

The project is in an initial development phase. Security fixes are applied to the
latest tagged release and the default branch.

## Reporting a Vulnerability

Please use GitHub's private security advisory feature for this repository. Do not
open a public issue containing credentials, exploit details, private datasets, or
sensitive run artifacts.

Include:

- the affected version or commit;
- a minimal reproduction;
- the expected impact;
- whether external services or untrusted memory content are involved; and
- any suggested mitigation.

## Secret Handling

The repository must never contain live API keys. Service integrations read local
environment variables documented in `.env.example`; the strict `ArchitectureSpec`
does not contain credentials or endpoints. Keep values in a local secret store or
CI secret manager. If a credential is committed, revoke it immediately before
rewriting history.

## Untrusted Memory Content

Treat retrieved pages, stored memories, generated shortcuts, and model output as
untrusted input. Implementations must validate structured output, constrain any
generated executable content, avoid unsafe deserialization, and fail closed at
security-sensitive gates.
