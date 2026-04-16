from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
import uvicorn
import threading
from fastmcp import FastMCP
import os
import json
import base64
import hmac
import hashlib
import time
import struct
from typing import Optional, Any

mcp = FastMCP("jwt-simple")


def _base64url_encode(data: bytes) -> str:
    """Encode bytes to base64url without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')


def _base64url_decode(data: str) -> bytes:
    """Decode base64url string (with or without padding) to bytes."""
    padding = 4 - len(data) % 4
    if padding != 4:
        data += '=' * padding
    return base64.urlsafe_b64decode(data)


def _get_hmac_algorithm(algorithm: str):
    """Return the hashlib algorithm for the given JWT HMAC algorithm name."""
    mapping = {
        "HS256": hashlib.sha256,
        "HS384": hashlib.sha384,
        "HS512": hashlib.sha512,
    }
    return mapping.get(algorithm)


@mcp.tool()
def create_hs256_token(
    secret: str,
    subject: Optional[str] = None,
    issuer: Optional[str] = None,
    audience: Optional[str] = None,
    expires_in_seconds: Optional[int] = 3600,
    not_before_seconds: Optional[int] = None,
    custom_claims: Optional[str] = None,
    algorithm: str = "HS256"
) -> dict:
    """
    Create a signed JWT token using HMAC symmetric algorithms (HS256, HS384, HS512).
    This mirrors the jwt-simple Rust library's symmetric key token creation.

    Args:
        secret: The HMAC secret key (string). Will be UTF-8 encoded.
        subject: Optional 'sub' claim.
        issuer: Optional 'iss' claim.
        audience: Optional 'aud' claim.
        expires_in_seconds: Token expiry duration in seconds from now (default 3600). Set to None for no expiry.
        not_before_seconds: Optional 'nbf' offset in seconds from now.
        custom_claims: Optional JSON string of additional claims to embed (e.g. '{"role": "admin"}').
        algorithm: JWT algorithm to use: HS256, HS384, or HS512 (default HS256).

    Returns:
        A dict with 'token' (the signed JWT string) and 'claims' (the payload as a dict).
    """
    algorithm = algorithm.upper()
    hash_fn = _get_hmac_algorithm(algorithm)
    if hash_fn is None:
        return {"error": f"Unsupported algorithm '{algorithm}'. Supported: HS256, HS384, HS512"}

    now = int(time.time())

    header = {"alg": algorithm, "typ": "JWT"}
    header_b64 = _base64url_encode(json.dumps(header, separators=(',', ':')).encode())

    payload: dict[str, Any] = {"iat": now}
    if subject:
        payload["sub"] = subject
    if issuer:
        payload["iss"] = issuer
    if audience:
        payload["aud"] = audience
    if expires_in_seconds is not None:
        payload["exp"] = now + expires_in_seconds
    if not_before_seconds is not None:
        payload["nbf"] = now + not_before_seconds

    if custom_claims:
        try:
            extra = json.loads(custom_claims)
            if isinstance(extra, dict):
                payload.update(extra)
            else:
                return {"error": "custom_claims must be a JSON object string"}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid custom_claims JSON: {str(e)}"}

    payload_b64 = _base64url_encode(json.dumps(payload, separators=(',', ':')).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()

    sig = hmac.new(secret.encode('utf-8'), signing_input, hash_fn).digest()
    sig_b64 = _base64url_encode(sig)

    token = f"{header_b64}.{payload_b64}.{sig_b64}"
    return {"token": token, "claims": payload, "algorithm": algorithm}


@mcp.tool()
def verify_hs_token(
    token: str,
    secret: str,
    expected_issuer: Optional[str] = None,
    expected_audience: Optional[str] = None,
    validate_expiry: bool = True
) -> dict:
    """
    Verify and decode a JWT token signed with an HMAC algorithm (HS256, HS384, HS512).
    This mirrors the jwt-simple Rust library's symmetric key token verification.

    Args:
        token: The JWT token string to verify.
        secret: The HMAC secret key used to sign the token.
        expected_issuer: If provided, validate that 'iss' matches this value.
        expected_audience: If provided, validate that 'aud' matches this value.
        validate_expiry: Whether to check 'exp' and 'nbf' claims (default True).

    Returns:
        A dict with 'valid' (bool), 'claims' (dict if valid), and 'error' (string if invalid).
    """
    parts = token.split('.')
    if len(parts) != 3:
        return {"valid": False, "error": "Token must have 3 parts separated by dots"}

    header_b64, payload_b64, sig_b64 = parts

    try:
        header = json.loads(_base64url_decode(header_b64))
    except Exception as e:
        return {"valid": False, "error": f"Failed to decode header: {str(e)}"}

    algorithm = header.get("alg", "").upper()
    hash_fn = _get_hmac_algorithm(algorithm)
    if hash_fn is None:
        return {"valid": False, "error": f"Unsupported algorithm '{algorithm}' in token header"}

    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected_sig = hmac.new(secret.encode('utf-8'), signing_input, hash_fn).digest()

    try:
        provided_sig = _base64url_decode(sig_b64)
    except Exception:
        return {"valid": False, "error": "Invalid signature encoding"}

    if not hmac.compare_digest(expected_sig, provided_sig):
        return {"valid": False, "error": "Signature verification failed"}

    try:
        claims = json.loads(_base64url_decode(payload_b64))
    except Exception as e:
        return {"valid": False, "error": f"Failed to decode payload: {str(e)}"}

    now = int(time.time())

    if validate_expiry:
        exp = claims.get("exp")
        if exp is not None and now > exp:
            return {"valid": False, "error": f"Token expired at {exp} (now: {now})"}

        nbf = claims.get("nbf")
        if nbf is not None and now < nbf:
            return {"valid": False, "error": f"Token not yet valid (nbf: {nbf}, now: {now})"}

    if expected_issuer:
        if claims.get("iss") != expected_issuer:
            return {"valid": False, "error": f"Issuer mismatch: expected '{expected_issuer}', got '{claims.get('iss')}'"}

    if expected_audience:
        aud = claims.get("aud")
        if isinstance(aud, list):
            if expected_audience not in aud:
                return {"valid": False, "error": f"Audience mismatch: '{expected_audience}' not in {aud}"}
        elif aud != expected_audience:
            return {"valid": False, "error": f"Audience mismatch: expected '{expected_audience}', got '{aud}'"}

    return {"valid": True, "claims": claims, "algorithm": algorithm}


@mcp.tool()
def decode_token_header(token: str) -> dict:
    """
    Decode and inspect the JWT token header without verification.
    Useful for peeking at the algorithm and key ID before deciding how to verify.
    This mirrors jwt-simple's ability to peek at token metadata before verification.

    Args:
        token: The JWT token string.

    Returns:
        A dict containing the decoded header fields.
    """
    parts = token.split('.')
    if len(parts) < 2:
        return {"error": "Invalid token format: must have at least 2 parts"}

    try:
        header = json.loads(_base64url_decode(parts[0]))
        return {"header": header}
    except Exception as e:
        return {"error": f"Failed to decode header: {str(e)}"}


@mcp.tool()
def decode_token_claims_unverified(token: str) -> dict:
    """
    Decode and inspect JWT token claims WITHOUT verifying the signature.
    WARNING: This does not verify the token's authenticity. Use only for inspection.
    This mirrors jwt-simple's peeking capability for metadata inspection.

    Args:
        token: The JWT token string.

    Returns:
        A dict containing 'header', 'claims', and a 'warning' about unverified status.
    """
    parts = token.split('.')
    if len(parts) != 3:
        return {"error": "Invalid token format: must have exactly 3 parts"}

    try:
        header = json.loads(_base64url_decode(parts[0]))
    except Exception as e:
        return {"error": f"Failed to decode header: {str(e)}"}

    try:
        claims = json.loads(_base64url_decode(parts[1]))
    except Exception as e:
        return {"error": f"Failed to decode claims: {str(e)}"}

    now = int(time.time())
    exp = claims.get("exp")
    nbf = claims.get("nbf")
    iat = claims.get("iat")

    time_info = {}
    if exp is not None:
        time_info["expires_at"] = exp
        time_info["expired"] = now > exp
        time_info["expires_in_seconds"] = exp - now
    if nbf is not None:
        time_info["not_before"] = nbf
        time_info["active"] = now >= nbf
    if iat is not None:
        time_info["issued_at"] = iat
        time_info["age_seconds"] = now - iat

    return {
        "warning": "UNVERIFIED - signature was NOT checked. Do not trust these claims for authentication.",
        "header": header,
        "claims": claims,
        "time_info": time_info
    }


@mcp.tool()
def explain_jwt_algorithms() -> dict:
    """
    Get an explanation of all JWT algorithms supported by the jwt-simple Rust library,
    including their use cases, security properties, and implementation guidance.

    Returns:
        A dict with algorithm categories and detailed descriptions.
    """
    return {
        "library": "jwt-simple (Rust crate)",
        "crate_url": "https://crates.io/crates/jwt-simple",
        "docs_url": "https://docs.rs/jwt-simple/",
        "algorithm_categories": {
            "symmetric_hmac": {
                "description": "Symmetric algorithms using a shared secret. Both signing and verification use the same key.",
                "use_case": "Best for single-service or microservice-internal tokens where you control both issuer and verifier.",
                "algorithms": {
                    "HS256": "HMAC-SHA-256. Most widely used. 256-bit security.",
                    "HS384": "HMAC-SHA-384. Stronger variant.",
                    "HS512": "HMAC-SHA-512. Strongest HMAC variant.",
                    "BLAKE2B": "BLAKE2B-256. Fast, modern hash function."
                },
                "rust_example": "let key = HS256Key::generate();\nlet claims = Claims::create(Duration::from_hours(2));\nlet token = key.authenticate(claims)?;"
            },
            "asymmetric_rsa": {
                "description": "Asymmetric RSA algorithms. Private key signs, public key verifies.",
                "use_case": "Best when multiple services need to verify tokens but only one service should issue them.",
                "algorithms": {
                    "RS256": "RSA + PKCS#1v1.5 padding + SHA-256. Widely compatible.",
                    "RS384": "RSA + PKCS#1v1.5 padding + SHA-384.",
                    "RS512": "RSA + PKCS#1v1.5 padding + SHA-512.",
                    "PS256": "RSA + PSS padding + SHA-256. More secure padding than RS256.",
                    "PS384": "RSA + PSS padding + SHA-384.",
                    "PS512": "RSA + PSS padding + SHA-512."
                },
                "rust_example": "let kp = RS256KeyPair::generate(2048)?;\nlet claims = Claims::create(Duration::from_hours(2));\nlet token = kp.sign(claims)?;\nlet pk = kp.public_key();\nlet verified = pk.verify_token::<NoCustomClaims>(&token, None)?;"
            },
            "asymmetric_ec": {
                "description": "Asymmetric Elliptic Curve algorithms. Smaller keys than RSA, equivalent security.",
                "use_case": "Modern replacement for RSA with better performance and smaller key sizes.",
                "algorithms": {
                    "ES256": "ECDSA on P-256 curve + SHA-256. Very widely deployed.",
                    "ES384": "ECDSA on P-384 curve + SHA-384.",
                    "ES256K": "ECDSA on secp256k1 curve (Bitcoin curve) + SHA-256."
                },
                "rust_example": "let kp = ES256KeyPair::generate();\nlet claims = Claims::create(Duration::from_hours(2));\nlet token = kp.sign(claims)?;"
            },
            "asymmetric_eddsa": {
                "description": "Edwards-curve Digital Signature Algorithm. Fast, safe, modern.",
                "use_case": "Best choice for new projects needing asymmetric JWT signing.",
                "algorithms": {
                    "EdDSA": "Ed25519 curve. Fast, secure, no side-channel vulnerabilities."
                },
                "rust_example": "let kp = Ed25519KeyPair::generate();\nlet claims = Claims::create(Duration::from_hours(2));\nlet token = kp.sign(claims)?;"
            }
        },
        "claims_fields": {
            "iss": "Issuer - who issued the token",
            "sub": "Subject - who the token is about",
            "aud": "Audience - who the token is intended for",
            "exp": "Expiration Time - Unix timestamp after which token is invalid",
            "nbf": "Not Before - Unix timestamp before which token is invalid",
            "iat": "Issued At - Unix timestamp when token was issued",
            "jti": "JWT ID - unique identifier for the token (replay attack prevention)"
        },
        "security_recommendations": [
            "Always validate expiry (exp claim) in production",
            "Use short-lived tokens (minutes to hours, not days)",
            "For new projects, prefer EdDSA (Ed25519) or ES256 over RSA",
            "Use audience (aud) claims to prevent token reuse across services",
            "Store secrets securely (environment variables, secret managers)",
            "Never use the 'none' algorithm",
            "jwt-simple explicitly rejects the 'none' algorithm"
        ]
    }


@mcp.tool()
def generate_hmac_secret(algorithm: str = "HS256") -> dict:
    """
    Generate a cryptographically secure random secret key suitable for use with
    HMAC JWT algorithms (HS256, HS384, HS512). The secret is returned as a
    hex string and base64url string for easy storage.

    Args:
        algorithm: The HMAC algorithm to generate a key for: HS256, HS384, or HS512.

    Returns:
        A dict with the generated secret in multiple encodings and usage guidance.
    """
    import secrets as secrets_module

    algorithm = algorithm.upper()
    key_sizes = {
        "HS256": 32,
        "HS384": 48,
        "HS512": 64,
    }

    key_size = key_sizes.get(algorithm)
    if key_size is None:
        return {"error": f"Unsupported algorithm '{algorithm}'. Supported: HS256, HS384, HS512"}

    raw_bytes = secrets_module.token_bytes(key_size)

    return {
        "algorithm": algorithm,
        "key_size_bytes": key_size,
        "key_size_bits": key_size * 8,
        "secret_hex": raw_bytes.hex(),
        "secret_base64url": _base64url_encode(raw_bytes),
        "secret_base64": base64.b64encode(raw_bytes).decode(),
        "rust_usage": f"let key = HS256Key::from_bytes(&hex::decode(\"<secret_hex>\").unwrap());",
        "warning": "Store this secret securely. Never commit it to version control. Use environment variables or a secrets manager."
    }


@mcp.tool()
def create_token_with_jti(
    secret: str,
    subject: Optional[str] = None,
    expires_in_seconds: int = 3600,
    algorithm: str = "HS256"
) -> dict:
    """
    Create a JWT token with a unique JWT ID (jti claim) for replay attack prevention.
    This mirrors jwt-simple's Mitigations against replay attacks feature.

    Args:
        secret: The HMAC secret key.
        subject: Optional 'sub' claim.
        expires_in_seconds: Token expiry in seconds (default 3600).
        algorithm: JWT algorithm: HS256, HS384, or HS512.

    Returns:
        A dict with 'token', 'jti' (the unique ID), and 'claims'.
    """
    import secrets as secrets_module

    algorithm = algorithm.upper()
    hash_fn = _get_hmac_algorithm(algorithm)
    if hash_fn is None:
        return {"error": f"Unsupported algorithm '{algorithm}'. Supported: HS256, HS384, HS512"}

    now = int(time.time())
    jti = secrets_module.token_hex(16)

    header = {"alg": algorithm, "typ": "JWT"}
    header_b64 = _base64url_encode(json.dumps(header, separators=(',', ':')).encode())

    payload: dict[str, Any] = {
        "iat": now,
        "exp": now + expires_in_seconds,
        "jti": jti
    }
    if subject:
        payload["sub"] = subject

    payload_b64 = _base64url_encode(json.dumps(payload, separators=(',', ':')).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()
    sig = hmac.new(secret.encode('utf-8'), signing_input, hash_fn).digest()
    sig_b64 = _base64url_encode(sig)

    token = f"{header_b64}.{payload_b64}.{sig_b64}"
    return {
        "token": token,
        "jti": jti,
        "claims": payload,
        "algorithm": algorithm,
        "replay_protection_note": "Store the jti in a cache/database with the exp time. Reject any token whose jti you've seen before."
    }


@mcp.tool()
def compare_token_algorithms() -> dict:
    """
    Get a comparison of JWT signing algorithms supported by jwt-simple to help
    choose the right algorithm for your use case.

    Returns:
        A detailed comparison table and recommendations.
    """
    return {
        "comparison_table": [
            {
                "algorithm": "HS256",
                "type": "Symmetric",
                "key_type": "Shared Secret",
                "key_size": "256 bits",
                "performance": "Fastest",
                "security_level": "128-bit",
                "compatibility": "Universal",
                "best_for": "Single-service tokens, simple setups"
            },
            {
                "algorithm": "HS512",
                "type": "Symmetric",
                "key_type": "Shared Secret",
                "key_size": "512 bits",
                "performance": "Fast",
                "security_level": "256-bit",
                "compatibility": "Universal",
                "best_for": "High-security symmetric tokens"
            },
            {
                "algorithm": "RS256",
                "type": "Asymmetric",
                "key_type": "RSA Key Pair",
                "key_size": "2048-4096 bits",
                "performance": "Slow (sign), Moderate (verify)",
                "security_level": "112-128 bit",
                "compatibility": "Universal (widely supported)",
                "best_for": "Legacy compatibility, OAuth2/OIDC"
            },
            {
                "algorithm": "ES256",
                "type": "Asymmetric",
                "key_type": "EC Key Pair (P-256)",
                "key_size": "256 bits",
                "performance": "Fast",
                "security_level": "128-bit",
                "compatibility": "Good (most modern systems)",
                "best_for": "Modern multi-service architectures"
            },
            {
                "algorithm": "EdDSA",
                "type": "Asymmetric",
                "key_type": "Ed25519 Key Pair",
                "key_size": "256 bits",
                "performance": "Fastest asymmetric",
                "security_level": "128-bit",
                "compatibility": "Growing (newer systems)",
                "best_for": "New projects, best security properties"
            }
        ],
        "decision_guide": {
            "single_service": "Use HS256 or HS512",
            "multiple_services_verify": "Use ES256 or EdDSA",
            "legacy_oauth_oidc": "Use RS256",
            "maximum_performance": "Use HS256 (symmetric) or EdDSA (asymmetric)",
            "maximum_compatibility": "Use HS256 or RS256",
            "new_greenfield_project": "Use EdDSA (Ed25519)"
        },
        "jwt_simple_rust_note": "The jwt-simple crate supports all these algorithms natively. Import with 'use jwt_simple::prelude::*;' to get access to all key types and claim builders."
    }




_SERVER_SLUG = "jedisct1-rust-jwt-simple"

def _track(tool_name: str, ua: str = ""):
    try:
        import urllib.request, json as _json
        data = _json.dumps({"slug": _SERVER_SLUG, "event": "tool_call", "tool": tool_name, "user_agent": ua}).encode()
        req = urllib.request.Request("https://www.volspan.dev/api/analytics/event", data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=1)
    except Exception:
        pass

async def health(request):
    return JSONResponse({"status": "ok", "server": mcp.name})

async def tools(request):
    registered = await mcp.list_tools()
    tool_list = [{"name": t.name, "description": t.description or ""} for t in registered]
    return JSONResponse({"tools": tool_list, "count": len(tool_list)})

sse_app = mcp.http_app(transport="sse")

app = Starlette(
    routes=[
        Route("/health", health),
        Route("/tools", tools),
        Mount("/", sse_app),
    ],
    lifespan=sse_app.lifespan,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
