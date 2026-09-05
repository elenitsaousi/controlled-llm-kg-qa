"""Fuseki connection helpers shared by the app and evaluation scripts."""

from __future__ import annotations

import base64
import os
from typing import Optional, Tuple

from rdflib.plugins.stores.sparqlstore import SPARQLStore


def fuseki_auth() -> Optional[Tuple[str, str]]:
    """Return optional basic-auth credentials for Fuseki.

    Local development often runs Fuseki without authentication, while deployed
    environments may protect the endpoint. Keeping this in one place avoids
    diverging behavior between the Streamlit app and batch evaluation scripts.
    """

    user = (
        os.getenv("FUSEKI_USERNAME")
        or os.getenv("FUSEKI_USER")
        or os.getenv("FUSEKI_API_USER")
        or ""
    ).strip()
    password = (
        os.getenv("FUSEKI_PASSWORD")
        or os.getenv("FUSEKI_PASS")
        or os.getenv("FUSEKI_API_PASSWORD")
        or ""
    ).strip()
    if not user or not password:
        return None
    return user, password


def fuseki_authorization_header() -> Optional[str]:
    auth = fuseki_auth()
    if not auth:
        return None
    token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def make_sparql_store(query_endpoint: str) -> SPARQLStore:
    auth = fuseki_auth()
    if auth:
        return SPARQLStore(query_endpoint.strip(), auth=auth)
    return SPARQLStore(query_endpoint.strip())
