def check_connectivity_to_server(server_location: ServerNetworkLocation, network_configuration: ServerNetworkConfiguration) -> ServerTlsProbingResult:

def get_preconfigured_tls_connection(self, override_tls_version: Optional[TlsVersionEnum] = None, ca_certificates_path: Optional[Path] = None, should_use_legacy_openssl: Optional[bool] = None, should_enable_server_name_indication: bool = True) -> SslConnection:

def _detect_support_for_tls_1_3(server_location: ServerNetworkLocation, network_config: ServerNetworkConfiguration) -> _TlsVersionDetectionResult:

def _detect_support_for_tls_1_2_or_below(server_location: ServerNetworkLocation, network_config: ServerNetworkConfiguration, tls_version: TlsVersionEnum) -> _TlsVersionDetectionResult:

def _detect_client_auth_requirement_with_tls_1_3(server_location: ServerNetworkLocation, network_config: ServerNetworkConfiguration) -> ClientAuthRequirementEnum:

def _detect_client_auth_requirement_with_tls_1_2_or_below(server_location: ServerNetworkLocation, network_config: ServerNetworkConfiguration, tls_version: TlsVersionEnum, cipher_list: str) -> ClientAuthRequirementEnum:

def _detect_ecdh_support(server_location: ServerNetworkLocation, network_config: ServerNetworkConfiguration, tls_version: TlsVersionEnum) -> bool:

def enable_ecdh_cipher_suites(tls_version: TlsVersionEnum, ssl_client: SslClient) -> None:
