"""True C2PA provenance signing for synthesized audio.

This is the *cryptographic provenance* layer (real C2PA manifests), complementing
the spectral watermark in ``c2pa_watermark`` (which is a robust in-signal mark that
survives re-encoding). Here we embed a standards-compliant C2PA manifest into the
output file declaring it AI-generated, using the IPTC ``trainedAlgorithmicMedia``
digital-source-type — the spec's canonical tag for synthetic media — signed with an
ES256 credential.

Trust note: the demo ships a *self-signed* ES256 certificate (auto-generated on
first use under ``$VG_C2PA_DIR`` or ``~/.voiceguard/c2pa``). The manifest is
cryptographically valid and machine-verifiable; the signer is simply not chained to
a public C2PA trust list. For production you would swap in a CA-issued cert via
``VG_C2PA_CERT`` / ``VG_C2PA_KEY`` (no code change).

Gracefully degrades: if the ``c2pa`` package is unavailable the API skips C2PA
signing and keeps the spectral watermark, so synthesis never hard-fails.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# IPTC digital-source-type for content created by a trained algorithm (the C2PA /
# CAI standard marker for AI-generated media).
TRAINED_ALGORITHMIC_MEDIA = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"


def _c2pa_dir() -> Path:
    d = Path(os.environ.get("VG_C2PA_DIR", str(Path.home() / ".voiceguard" / "c2pa")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_available() -> bool:
    """True if the c2pa runtime is importable."""
    try:
        import c2pa  # noqa: F401

        return True
    except Exception:
        return False


def _ensure_credentials() -> tuple[Path, Path]:
    """Return (cert_pem, key_pkcs8_pem), generating a self-signed ES256 demo
    credential on first use. Overridable via VG_C2PA_CERT / VG_C2PA_KEY."""
    env_cert, env_key = os.environ.get("VG_C2PA_CERT"), os.environ.get("VG_C2PA_KEY")
    if env_cert and env_key:
        return Path(env_cert), Path(env_key)

    d = _c2pa_dir()
    cert_p, key_p = d / "cert.pem", d / "key.pem"
    if cert_p.exists() and key_p.exists():
        return cert_p, key_p

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "VoiceGuard Demo Signer"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "VoiceGuard"),
        ]
    )
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        # Profile matched to what c2pa-rs accepts (non-critical KU/BC, SKI+AKI present).
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=False)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=False,
        )
        # C2PA cert profile requires the emailProtection EKU for document signing.
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_p.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,  # c2pa requires PKCS#8, not SEC1
            serialization.NoEncryption(),
        )
    )
    key_p.chmod(0o600)
    return cert_p, key_p


def _make_signer():
    import c2pa
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, utils

    cert_p, key_p = _ensure_credentials()
    key = serialization.load_pem_private_key(key_p.read_bytes(), password=None)

    def _sign(data: bytes) -> bytes:
        der = key.sign(data, ec.ECDSA(hashes.SHA256()))
        r, s = utils.decode_dss_signature(der)
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")  # COSE raw r||s

    # tsa_url=None -> offline signing (no timestamp authority needed)
    return c2pa.Signer.from_callback(_sign, c2pa.C2paSigningAlg.ES256, cert_p.read_text(), None)


def sign_file(
    in_path: str,
    out_path: str,
    *,
    software_agent: str = "VoiceGuard",
    title: str = "VoiceGuard synthesized audio",
    claim_generator: str = "VoiceGuard",
    version: str = "1.0.0",
) -> dict:
    """Embed a signed C2PA manifest declaring AI-generated provenance.

    Returns a summary dict ``{signed, reason?, manifest_label?}``. Never raises for
    expected-missing-dependency cases — callers can treat ``signed=False`` as
    "spectral watermark only".
    """
    if not is_available():
        return {"signed": False, "reason": "c2pa runtime not installed"}
    try:
        import c2pa

        manifest = {
            "claim_generator_info": [{"name": claim_generator, "version": version}],
            "title": title,
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.created",
                                "digitalSourceType": TRAINED_ALGORITHMIC_MEDIA,
                                "softwareAgent": software_agent,
                            }
                        ]
                    },
                }
            ],
        }
        signer = _make_signer()
        builder = c2pa.Builder.from_json(manifest)
        builder.sign_file(in_path, out_path, signer)
        return {"signed": True, "digital_source_type": "trainedAlgorithmicMedia"}
    except Exception as exc:  # noqa: BLE001 — provenance is best-effort, never fatal
        logger.warning("C2PA signing failed for %s: %s", in_path, exc, exc_info=True)
        return {"signed": False, "reason": f"{type(exc).__name__}: {exc}"}


def verify_file(path: str) -> dict:
    """Read back any embedded C2PA manifest. Returns
    ``{has_manifest, validation_state?, ai_generated?, software_agent?}``."""
    if not is_available():
        return {"has_manifest": False, "reason": "c2pa runtime not installed"}
    try:
        import c2pa

        mime = "audio/wav" if str(path).lower().endswith(".wav") else "audio/mpeg"
        with open(path, "rb") as f:
            reader = c2pa.Reader(mime, f)
            data = json.loads(reader.json())
        active = data.get("active_manifest")
        if not active or active not in data.get("manifests", {}):
            return {"has_manifest": False}
        man = data["manifests"][active]
        ai, agent = False, None
        for a in man.get("assertions", []):
            if a.get("label", "").startswith("c2pa.actions"):
                for act in a.get("data", {}).get("actions", []):
                    if act.get("digitalSourceType", "").endswith("trainedAlgorithmicMedia"):
                        ai = True
                        agent = act.get("softwareAgent")
        return {
            "has_manifest": True,
            "validation_state": data.get("validation_state"),
            "ai_generated": ai,
            "software_agent": agent,
            "manifest_label": active,
        }
    except Exception as exc:  # noqa: BLE001
        return {"has_manifest": False, "reason": f"{type(exc).__name__}: {exc}"}
