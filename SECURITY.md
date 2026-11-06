# Security

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private vulnerability reporting for this
repository (Security tab, "Report a vulnerability").

## API keys

PCBRouteAgent reads API keys from `~/.config/routeagent/env` (or `$ROUTEAGENT_ENV`, or a gitignored
`./.env`), never from the repository. Only key names are ever printed. If a key is committed by
mistake, revoke it with the provider first, then remove it from the history.
