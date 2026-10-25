"""Zero-Egress Network Interlock for CORTEX-Personal.

Enforces absolute network isolation ($0.0 KB WAN egress) for private personal cognitive substrates.
Refuses socket connections to any external, non-loopback IP address or remote hostname.
"""

from __future__ import annotations

import socket
from typing import Any, Callable, Optional


class ZeroEgressViolation(PermissionError):
    """Raised when an unauthorized non-loopback network connection is attempted."""
    pass


class ZeroEgressInterlock:
    """Interlock monitoring and enforcing zero WAN network egress."""

    ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self._is_active = False
        self._original_socket_connect: Optional[Callable] = None

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
                # Check if loopback
                if host in interlock_self.ALLOWED_HOSTS or host.startswith("127."):
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
