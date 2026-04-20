# Privacy

*Last updated: 2026-04-20*

This page describes what `cognilateral.com` collects, how long it keeps it, and what you can ask to have deleted.

## If you only use the library (`pip install cognilateral-trust`)

I collect nothing. The library runs on your machine, talks to no network services, has zero runtime dependencies. PyPI logs downloads; that is between you and PyPI.

## If you use the hosted API (`cognilateral.com/api/v1/evaluate`)

### What I collect

- **Your email** — captured when you signed up for an API key.
- **Your API key** — stored as a hash. I cannot recover the plaintext after creation.
- **Each API request** — timestamp, tier result, response time, and the arguments you sent: the numeric `confidence` and the boolean flags `is_reversible` and `touches_external`.
- **The `context` field** — logged verbatim for debugging and accountability. **Do not put PII, secrets, or regulated data in `context`.** If you do, I will delete on request, but I cannot guarantee removal from existing backups.
- **Your IP address** — logged per request for rate limiting and abuse prevention.

### What I do not collect

- No third-party analytics, no advertising pixels, no cookies beyond your login session.
- No model outputs, prompts, or training data. You send a number and some flags — that is the entire request payload.
- No tracking across sites.

### Who sees it

- **The maintainer** (Eric Mumford), for operating the service.
- **Stripe**, if you are on a paid tier. Stripe handles payment data directly; I never see your card number. See Stripe's privacy policy for their handling.
- Nobody else. No analytics vendors, no data brokers, no training partnerships.

### How long I keep it

| Data | Retention |
|------|-----------|
| API request logs | 90 days, then aggregated to counts and deleted |
| Your email and API key | Until you ask me to delete, or 12 months after your last call if you do not return |
| Stripe transaction records | Per Stripe's retention policy |

### What you can ask for

Email `eric@cognilateral.com` with subject `[DATA REQUEST]` and specify one of:

- **Export** — I will send you everything I have on you within 30 days.
- **Delete** — I will delete your email, key, and request logs within 30 days. Stripe transaction records are retained per their legal requirements.
- **Correct** — I will fix any data you flag as wrong.

You do not need to give a reason. I will not make you fill out a form.

## Jurisdiction

I operate from the United States. If you are in the EU, UK, or California, the rights listed above are intended to align with GDPR, UK-GDPR, and CCPA/CPRA baselines. This is a good-faith engineer's privacy page, not legal advice; specific jurisdictional requirements should be reviewed with counsel for enterprise use. If a specific right in your jurisdiction is missing, email me and I will add it.

## Changes to this page

If this page changes, every active API key holder gets an email at least 30 days before any change that expands what I collect. You always see the "Last updated" date at the top.

## Contact

`eric@cognilateral.com`
