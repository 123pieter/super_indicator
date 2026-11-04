"""Provider construction and validation; imports make no network requests."""
from .provider_polygon import PolygonProvider
from .provider_hyperliquid import HyperliquidProvider

PROVIDERS = ("polygon", "hyperliquid")


def validate_provider(name):
    name = name.lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}")
    return name


def get_provider(name, api_key=None):
    name = validate_provider(name)
    if name == "polygon":
        return PolygonProvider(api_key)
    from config import HYPERLIQUID_DEXES
    return HyperliquidProvider(dexes=HYPERLIQUID_DEXES)
