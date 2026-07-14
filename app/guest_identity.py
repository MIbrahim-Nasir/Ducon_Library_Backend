"""
Guest identity signals — client IP, Cloudflare metadata, composite server hash.

Composite signals supplement the client fingerprint (X-Guest-Fingerprint) for
abuse auditing and optional shared caps. They are NOT used to merge different
session UUIDs into one quota row — fingerprint_hash remains per-device; subnet
pooling is opt-in only via GUEST_SUBNET_LIMIT.

Forwarded IP headers (CF-Connecting-IP, X-Forwarded-For, …) are only trusted
when the immediate peer is in TRUSTED_PROXY_CIDRS (default: loopback). Set this
to your Apache / Cloudflare tunnel peer ranges in production.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from ipaddress import ip_address, ip_network, ip_address as parse_ip
from typing import Mapping, Optional

from app.hashing import sha256_hex

# Cloudflare header aliases (HTTP header names are case-insensitive).
_CF_CONNECTING_IP = ("cf-connecting-ip", "CF-Connecting-IP")
_TRUE_CLIENT_IP = ("true-client-ip", "True-Client-IP")
_CF_JA4 = ("cf-ja4", "cf-ja4-fingerprint", "CF-JA4-Fingerprint")
_CF_JA3 = ("cf-ja3-hash", "CF-JA3-Hash")
_CF_ASN = ("cf-ipasn", "cf-ip-asn", "CF-IPASN", "cf-asn")
_CF_COUNTRY = ("cf-ipcountry", "CF-IPCountry")


@dataclass(frozen=True)
class GuestServerSignals:
    """Server-side signals collected on each guest request."""

    ja4_fingerprint: Optional[str] = None
    ja3_hash: Optional[str] = None
    asn: Optional[str] = None
    country: Optional[str] = None
    accept_language: Optional[str] = None
    subnet_key: Optional[str] = None


@dataclass(frozen=True)
class GuestRequestIdentity:
    """Resolved guest identity for session lookup and abuse auditing."""

    client_ip: str
    ip_hash: str
    fingerprint_hash: Optional[str]  # already server-normalised (double-hashed)
    composite_hash: Optional[str]
    subnet_key: Optional[str]
    ja4_fingerprint: Optional[str]
    asn: Optional[str]


def _header_value(headers: Mapping[str, str], names: tuple[str, ...]) -> Optional[str]:
    for name in names:
        value = headers.get(name)
        if value is None:
            # Starlette lowercases keys; try lowercase variant.
            value = headers.get(name.lower())
        if value and value.strip():
            return value.strip()
    return None


@lru_cache(maxsize=1)
def _trusted_proxy_networks():
    raw = os.getenv("TRUSTED_PROXY_CIDRS", "127.0.0.1/32,::1/128")
    nets = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            nets.append(ip_network(part, strict=False))
        except ValueError:
            continue
    return tuple(nets)


def peer_is_trusted_proxy(peer_host: Optional[str]) -> bool:
    """True when the immediate TCP peer may supply forwarded-client headers."""
    if not peer_host:
        return False
    # Allow opting into "trust all peers" for misconfigured hosts (not recommended).
    if os.getenv("TRUST_FORWARDED_IP_HEADERS", "").lower() in ("1", "true", "yes", "all"):
        return True
    try:
        addr = parse_ip(peer_host.strip())
    except ValueError:
        return False
    return any(addr in net for net in _trusted_proxy_networks())


def extract_client_ip(
    headers: Mapping[str, str],
    *,
    peer_host: Optional[str] = None,
) -> str:
    """Return the best-effort client IP.

    Forwarded headers are honored only when ``peer_host`` is a trusted proxy
    (see TRUSTED_PROXY_CIDRS). Otherwise the peer address is used so clients
    cannot spoof CF-Connecting-IP / X-Forwarded-For.
    """
    if peer_is_trusted_proxy(peer_host):
        cf_ip = _header_value(headers, _CF_CONNECTING_IP)
        if cf_ip:
            return cf_ip

        true_client = _header_value(headers, _TRUE_CLIENT_IP)
        if true_client:
            return true_client

        xff = _header_value(headers, ("x-forwarded-for", "X-Forwarded-For"))
        if xff:
            # First hop is the original client when proxied through CDNs.
            first = xff.split(",")[0].strip()
            if first:
                return first

    if peer_host:
        return peer_host
    return "unknown"


def subnet_key(ip: str) -> Optional[str]:
    """Truncate client IP to IPv4 /24 or IPv6 /64 for supplementary abuse tracking."""
    if not ip or ip == "unknown":
        return None
    try:
        addr = ip_address(ip.strip())
    except ValueError:
        return None

    if addr.version == 4:
        net = ip_network(f"{addr}/24", strict=False)
        return f"ipv4:{net.network_address}/24"
    net = ip_network(f"{addr}/64", strict=False)
    return f"ipv6:{net.network_address}/64"


def extract_cloudflare_signals(
    headers: Mapping[str, str],
    *,
    client_ip: Optional[str] = None,
) -> GuestServerSignals:
    """Collect TLS / network metadata forwarded by Cloudflare (when present).

    When ``client_ip`` is omitted, prefers ``CF-Connecting-IP`` for subnet
    derivation (audit only). Rate-limit / quota identity must use
    ``extract_client_ip(..., peer_host=...)`` which enforces trusted proxies.
    """
    ip = client_ip or _header_value(headers, _CF_CONNECTING_IP) or "unknown"
    return GuestServerSignals(
        ja4_fingerprint=_header_value(headers, _CF_JA4),
        ja3_hash=_header_value(headers, _CF_JA3),
        asn=_header_value(headers, _CF_ASN),
        country=_header_value(headers, _CF_COUNTRY),
        accept_language=_header_value(headers, ("accept-language", "Accept-Language")),
        subnet_key=subnet_key(ip),
    )


def composite_server_hash(signals: GuestServerSignals) -> Optional[str]:
    """SHA-256 digest of server-side signals for secondary identity binding."""
    parts = [
        signals.ja4_fingerprint or signals.ja3_hash or "",
        signals.asn or "",
        signals.accept_language or "",
        signals.subnet_key or "",
    ]
    if not any(parts):
        return None
    return sha256_hex("|".join(parts))


def build_guest_request_identity(
    headers: Mapping[str, str],
    *,
    peer_host: Optional[str] = None,
    raw_fingerprint: Optional[str] = None,
) -> GuestRequestIdentity:
    """Resolve all guest identity fields from request / WebSocket headers."""
    from app.guest_usage import normalise_fingerprint_hash

    client_ip = extract_client_ip(headers, peer_host=peer_host)
    signals = extract_cloudflare_signals(headers, client_ip=client_ip)
    fp_hash = normalise_fingerprint_hash(raw_fingerprint)
    return GuestRequestIdentity(
        client_ip=client_ip,
        ip_hash=sha256_hex(client_ip),
        fingerprint_hash=fp_hash,
        composite_hash=composite_server_hash(signals),
        subnet_key=signals.subnet_key,
        ja4_fingerprint=signals.ja4_fingerprint or signals.ja3_hash,
        asn=signals.asn,
    )
