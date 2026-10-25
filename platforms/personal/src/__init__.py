"""CORTEX-Personal Platform: Privacy-first offline personal cognitive substrate."""

from .interlock import ZeroEgressInterlock, ZeroEgressViolation
from .substrate import PersonalSubstrateCoordinator, PersonalQueryResult
from .tenant_paging import PersonalTenantPagingEngine, TenantFactorSlice
from .vault import EncryptedPersonalVault

__all__ = [
    "PersonalSubstrateCoordinator",
    "PersonalQueryResult",
    "EncryptedPersonalVault",
    "ZeroEgressInterlock",
    "ZeroEgressViolation",
    "PersonalTenantPagingEngine",
    "TenantFactorSlice",
]
