"""Zero-Egress Network Interlock for CORTEX-Personal.

Enforces absolute network isolation ($0.0 KB WAN egress) for private personal cognitive substrates.
Refuses socket connections to any external, non-loopback IP address or remote hostname.
"""

from __future__ import annotations

import os
import socket
from typing import Any, Callable, Optional, Set


class ZeroEgressViolation(PermissionError):
    """Raised when an unauthorized non-loopback network connection is attempted."""
    pass


class ZeroEgressInterlock:
    """Interlock monitoring and enforcing zero WAN network egress."""

    DEFAULT_ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}

    def __init__(self, strict_mode: bool = True, allowed_hosts: Optional[Set[str]] = None):
        self.strict_mode = strict_mode
        self._is_active = False
        self._original_socket_connect: Optional[Callable] = None
        
        # Merge default hosts, custom allowed hosts, and CORTEX_ALLOWED_HOSTS environment variable
        self.allowed_hosts = set(self.DEFAULT_ALLOWED_HOSTS)
        if allowed_hosts:
            self.allowed_hosts.update(allowed_hosts)
        
        env_allowed = os.getenv("CORTEX_ALLOWED_HOSTS")
        if env_allowed:
            for h in env_allowed.split(","):
                clean_h = h.strip()
                if clean_h:
                    self.allowed_hosts.add(clean_h)

    # Backwards-compatibility property
    @property
    def ALLOWED_HOSTS(self) -> Set[str]:
        return self.allowed_hosts

    def enable(self) -> None:
        """Activate the zero-egress interlock at the process socket layer."""
        if self._is_active:
            return

        self._original_socket_connect = socket.socket.connect
        interlock_self = self

        def guarded_connect(sock: socket.socket, address: Any) -> Any:
            host = None
            if isinstance(address, tuple) and len(address) >= 1:
                host = str(address[0])
            elif isinstance(address, str):
                host = address

            if host is not None:
                # Check if loopback or explicitly permitted
                if host in interlock_self.allowed_hosts or host.startswith("127."):
                    return interlock_self._original_socket_connect(sock, address)

            # Block all non-loopback WAN/LAN connections
            msg = f"ZeroEgressInterlock: External network egress to {address!r} strictly blocked."
            if interlock_self.strict_mode:
                raise ZeroEgressViolation(msg)
            return None

        socket.socket.connect = guarded_connect
        self._is_active = True

    def disable(self) -> None:
        """Deactivate the socket interlock and restore default system socket behavior."""
        if not self._is_active:
            return

        if self._original_socket_connect:
            socket.socket.connect = self._original_socket_connect
            self._original_socket_connect = None
        self._is_active = False

    def is_active(self) -> bool:
        """Return whether zero-egress interlock is actively guarding the runtime."""
        return self._is_active

    def __enter__(self) -> ZeroEgressInterlock:
        self.enable()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disable()
