from dataclasses import dataclass
from typing import Optional, List, Union, cast
import cryptography
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509 import (
    ExtensionNotFound,
    ExtensionOID,
    Certificate,
    load_pem_x509_certificate,
    TLSFeature,
    DNSName,
    IPAddress,
)
from cryptography.x509.ocsp import (
    load_der_ocsp_response,
    OCSPResponseStatus,
    OCSPResponse,
)
import nassl.ocsp_response
from sslyze.plugins.certificate_info._symantec import SymantecDistructTester
from sslyze.plugins.certificate_info.trust_stores.trust_store import TrustStore, PathValidationResult

@dataclass(frozen=True)
class CertificateDeploymentAnalysisResult:
    """The result of analyzing a server's certificate to verify its validity.

    Any certificate available within the fields that follow is parsed as a ``Certificate`` object using the cryptography
    module; documentation is available at
    https://cryptography.io/en/latest/x509/reference.html?highlight=Certificate#cryptography.x509.Certificate

    Attributes:
        received_certificate_chain: The certificate chain sent by the server; index 0 is the leaf certificate.
        verified_certificate_chain: The verified certificate chain returned by OpenSSL for one of the trust stores
            packaged within SSLyze. Will be ``None`` if the validation failed with all of the available trust stores
            (Apple, Mozilla, etc.). This is essentially a shortcut to
            ``path_validation_result_list[0].verified_certificate_chain``.
        path_validation_results: The result of validating the server's
            certificate chain using each trust store that is packaged with SSLyze (Mozilla, Apple, etc.).
            If for a given trust store, the validation was successful, the verified certificate chain can be
             retrieved from the ``PathValidationResult``.
        leaf_certificate_is_ev: ``True`` if the leaf certificate is Extended Validation, according to Mozilla.
        leaf_certificate_has_must_staple_extension: ``True`` if the OCSP must-staple extension is present in the leaf
            certificate.
        leaf_certificate_signed_certificate_timestamps_count: The number of Signed Certificate
            Timestamps (SCTs) for Certificate Transparency embedded in the leaf certificate. ``None`` if the version of
            OpenSSL installed on the system is too old to be able to parse the SCT extension.
        received_chain_has_valid_order: ``True`` if the certificate chain returned by the server was sent in the right
            order. `None`` if any of the certificates in the chain could not be parsed.
        received_chain_contains_anchor_certificate: ``True`` if the server included the anchor/root
            certificate in the chain it sends back to clients. ``None`` if the verified chain could not be built.
        verified_chain_has_sha1_signature: ``True`` if any of the leaf or intermediate certificates are
            signed using the SHA-1 algorithm. ``None`` if the verified chain could not be built.
        verified_chain_has_legacy_symantec_anchor: ``True`` if the certificate chain contains a distrusted Symantec
            anchor
            (https://blog.qualys.com/ssllabs/2017/09/26/google-and-mozilla-deprecating-existing-symantec-certificates).
            ``None`` if the verified chain could not be built.
        ocsp_response: The OCSP response returned by the server. ``None`` if no response was sent by the server or if
            the scan was run through an HTTP proxy (the proxy will not forward the server's OCSP response). If present,
            the OCSP response is an ``OCSPResponse`` object parsed using the cryptography module; documentation is
            available at
            https://cryptography.io/en/latest/x509/ocsp.html?highlight=OCSPResponse#cryptography.x509.ocsp.OCSPResponse
        ocsp_response_is_trusted: ``True`` if the OCSP response is trusted using the Mozilla trust store.
            ``None`` if no OCSP response was sent by the server.

    """
    received_certificate_chain: List[Certificate]
    verified_certificate_chain: Optional[List[Certificate]]
    path_validation_results: List[PathValidationResult]
    leaf_certificate_is_ev: bool
    leaf_certificate_has_must_staple_extension: bool
    leaf_certificate_signed_certificate_timestamps_count: Optional[int]
    received_chain_has_valid_order: Optional[bool]
    received_chain_contains_anchor_certificate: Optional[bool]
    verified_chain_has_sha1_signature: Optional[bool]
    verified_chain_has_legacy_symantec_anchor: Optional[bool]
    ocsp_response: Optional[OCSPResponse]
    ocsp_response_is_trusted: Optional[bool]

    @property
    def verified_certificate_chain_as_pem(self) -> Optional[List[str]]:
        """Get one of the verified certificate chains if one was successfully built using any of the trust stores."""
        if self.verified_certificate_chain is None:
            return None
        pem_certs: List[str] = []
        for certificate in self.verified_certificate_chain:
            pem_certs.append(certificate.public_bytes(Encoding.PEM).decode('ascii'))
        return pem_certs

    @property
    def received_certificate_chain_as_pem(self) -> List[str]:
        pem_certs: List[str] = []
        for certificate in self.received_certificate_chain:
            pem_certs.append(certificate.public_bytes(Encoding.PEM).decode('ascii'))
        return pem_certs

class CertificateDeploymentAnalyzer:
    """Utility class for analyzing a certificate chain as deployed on a specific server.

    Useful for checking a server's certificate chain without having to use the CertificateInfoPlugin.
    """

    def __init__(
        self,
        server_subject: Union[DNSName, IPAddress],
        server_certificate_chain_as_pem: List[str],
        server_ocsp_response: Optional[nassl.ocsp_response.OCSPResponse],
        trust_stores_for_validation: List[TrustStore],
    ) -> None:
        self.server_subject: Union[DNSName, IPAddress] = server_subject
        self.server_certificate_chain_as_pem: List[str] = server_certificate_chain_as_pem
        self.server_ocsp_response: Optional[nassl.ocsp_response.OCSPResponse] = server_ocsp_response
        self.trust_stores_for_validation: List[TrustStore] = trust_stores_for_validation

    def perform(self) -> CertificateDeploymentAnalysisResult:
        received_certificate_chain: List[Certificate] = [
            load_pem_x509_certificate(pem_cert.encode('ascii'), backend=default_backend())
            for pem_cert in self.server_certificate_chain_as_pem
        ]
        leaf_cert: Certificate = received_certificate_chain[0]
        has_ocsp_must_staple: bool = False
        try:
            tls_feature_ext = leaf_cert.extensions.get_extension_for_oid(ExtensionOID.TLS_FEATURE)
            tls_feature_value: TLSFeature = cast(TLSFeature, tls_feature_ext.value)
            for feature_type in tls_feature_value:
                if feature_type == cryptography.x509.TLSFeatureType.status_request:
                    has_ocsp_must_staple = True
                    break
        except ExtensionNotFound:
            pass

        is_chain_order_valid: Optional[bool] = True
        previous_issuer: Optional[cryptography.x509.Name] = None
        for index, cert in enumerate(received_certificate_chain):
            try:
                current_subject: cryptography.x509.Name = cert.subject
            except ValueError:
                is_chain_order_valid = None
                break
            if index > 0:
                if current_subject != previous_issuer:
                    is_chain_order_valid = False
                    break
            try:
                previous_issuer = cert.issuer
            except KeyError:
                previous_issuer = None
            except ValueError:
                previous_issuer = None
                is_chain_order_valid = None
                break

        is_leaf_certificate_ev: bool = False
        for trust_store in self.trust_stores_for_validation:
            if trust_store.ev_oids is None:
                continue
            if trust_store.is_certificate_extended_validation(leaf_cert):
                is_leaf_certificate_ev = True
                break

        number_of_scts: Optional[int] = 0
        try:
            sct_ext = leaf_cert.extensions.get_extension_for_oid(ExtensionOID.PRECERT_SIGNED_CERTIFICATE_TIMESTAMPS)
            if isinstance(sct_ext.value, cryptography.x509.UnrecognizedExtension):
                number_of_scts = None
            else:
                sct_ext_value = cast(
                    cryptography.x509.PrecertificateSignedCertificateTimestamps, sct_ext.value
                )
                number_of_scts = len(sct_ext_value)
        except ExtensionNotFound:
            pass

        all_path_validation_results: List[PathValidationResult] = []
        for trust_store in self.trust_stores_for_validation:
            path_validation_result: PathValidationResult = trust_store.verify_certificate_chain(
                self.server_certificate_chain_as_pem, self.server_subject
            )
            all_path_validation_results.append(path_validation_result)

        trust_store_that_can_build_verified_chain: Optional[TrustStore] = None
        verified_certificate_chain: Optional[List[Certificate]] = None

        def sort_function(path_validation: PathValidationResult) -> str:
            return path_validation.trust_store.name.lower()

        all_path_validation_results.sort(key=sort_function)
        for path_validation_result in all_path_validation_results:
            if path_validation_result.was_validation_successful:
                trust_store_that_can_build_verified_chain = path_validation_result.trust_store
                verified_certificate_chain = path_validation_result.verified_certificate_chain
                break

        has_anchor_in_certificate_chain: Optional[bool] = None
        if verified_certificate_chain:
            has_anchor_in_certificate_chain = verified_certificate_chain[-1] in received_certificate_chain

        has_sha1_in_certificate_chain: Optional[bool] = None
        if verified_certificate_chain:
            has_sha1_in_certificate_chain = False
            for cert in verified_certificate_chain[:-1]:
                if isinstance(cert.signature_hash_algorithm, hashes.SHA1):
                    has_sha1_in_certificate_chain = True
                    break

        verified_chain_has_legacy_symantec_anchor: Optional[bool] = None
        if verified_certificate_chain:
            symantec_distrust_timeline = SymantecDistructTester.get_distrust_timeline(verified_certificate_chain)
            verified_chain_has_legacy_symantec_anchor = bool(symantec_distrust_timeline)

        is_ocsp_response_trusted: Optional[bool] = None
        final_ocsp_response: Optional[OCSPResponse] = None
        if self.server_ocsp_response:
            final_ocsp_response = load_der_ocsp_response(self.server_ocsp_response.as_der_bytes())
            if (
                trust_store_that_can_build_verified_chain
                and final_ocsp_response.response_status == OCSPResponseStatus.SUCCESSFUL
            ):
                try:
                    nassl.ocsp_response.verify_ocsp_response(
                        self.server_ocsp_response, trust_store_that_can_build_verified_chain.path
                    )
                    is_ocsp_response_trusted = True
                except nassl.ocsp_response.OcspResponseNotTrustedError:
                    is_ocsp_response_trusted = False

        return CertificateDeploymentAnalysisResult(
            received_certificate_chain=received_certificate_chain,
            verified_certificate_chain=verified_certificate_chain,
            path_validation_results=all_path_validation_results,
            leaf_certificate_is_ev=is_leaf_certificate_ev,
            leaf_certificate_has_must_staple_extension=has_ocsp_must_staple,
            leaf_certificate_signed_certificate_timestamps_count=number_of_scts,
            received_chain_has_valid_order=is_chain_order_valid,
            received_chain_contains_anchor_certificate=has_anchor_in_certificate_chain,
            verified_chain_has_sha1_signature=has_sha1_in_certificate_chain,
            verified_chain_has_legacy_symantec_anchor=verified_chain_has_legacy_symantec_anchor,
            ocsp_response=final_ocsp_response,
            ocsp_response_is_trusted=is_ocsp_response_trusted,
        )
