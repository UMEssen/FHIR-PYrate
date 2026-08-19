import json
import logging
import os
import pathlib
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable
from typing import Any

import requests

from fhir_pyrate.util import now_utc

logger = logging.getLogger(__name__)


class DeviceCodeAuthError(Exception):
    """Raised when the OAuth 2.0 Device Authorization Grant cannot be completed."""


def _json_body(response: requests.Response) -> dict[str, Any]:
    """Return the JSON object of a response, or an empty dict if the body is not a
    JSON object — so that a broken identity provider leads to a clear error about
    the missing field instead of an obscure AttributeError."""
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def print_device_prompt(
    verification_uri: str, user_code: str, verification_uri_complete: str | None
) -> None:
    """
    Print the device-flow sign-in instructions to stderr.

    This is the default prompt callback of :class:`DeviceCodeAuth`; pass your own
    callable with the same signature to display the instructions differently.

    :param verification_uri: The URL where the user can enter the user code
    :param user_code: The code that the user has to confirm or enter
    :param verification_uri_complete: The URL with the user code already embedded, if the
    server provides one
    """
    lines = ["", "To sign in, open the following URL in any browser:"]
    if verification_uri_complete is not None:
        lines.append(f"  {verification_uri_complete}")
        lines.append(f"and check that the page shows the code {user_code}.")
    else:
        lines.append(f"  {verification_uri}")
        lines.append(f"and enter the code {user_code}.")
    lines.append("Waiting for the sign-in to be approved...")
    print("\n".join(lines), file=sys.stderr, flush=True)


class DeviceCodeAuth(requests.auth.AuthBase):
    """
    Authenticate via the OAuth 2.0 Device Authorization Grant (RFC 8628), as offered
    e.g. by Keycloak, and keep the obtained access token fresh.

    On the first use, the class prints a URL (and a user code) that the user opens in any
    browser to approve the sign-in — the usual single sign-on of the identity provider
    applies, and no password ever goes through this class. The obtained access token is
    then attached to every request as a Bearer header, refreshed with the refresh token
    before it expires, and — if a request still comes back as unauthorized, or as a
    redirect to the identity provider's sign-in page (how proxies like oauth2-proxy
    answer an expired token) — refreshed once more and the request retried, in the
    same way as ``requests``' own digest authentication.

    In contrast to :class:`~fhir_pyrate.util.token_auth.TokenAuth`, this class does not
    rely on session hooks, so the same instance keeps working when it is copied to
    another :class:`requests.Session` (as :class:`~fhir_pyrate.pirate.Pirate` does for
    its caching session). Instances can also be pickled, so they survive the
    multiprocessing of :class:`~fhir_pyrate.pirate.Pirate` and
    :class:`~fhir_pyrate.dicom_downloader.DicomDownloader` (as long as a possible custom
    ``prompt_callback`` is a picklable function). One caveat applies: every worker
    process holds a copy of the same refresh token, so if the identity provider
    rotates refresh tokens on every use (e.g. Keycloak's "Revoke Refresh Token"
    option, off by default), the first worker's refresh invalidates the others'
    copies and each of them falls back to a new interactive sign-in. In that case
    use a single process, or make sure the access-token lifetime covers the run.

    :param client_id: The OAuth client ID to authenticate as (a public client; for a
    confidential client also pass ``client_secret``)
    :param issuer_url: The OpenID Connect issuer URL, e.g.
    ``https://keycloak.example.com/realms/example-realm``; the device authorization and
    token endpoints are then read from the issuer's well-known configuration. Either
    this or both ``device_authorization_url`` and ``token_url`` must be given.
    :param device_authorization_url: The device authorization endpoint, if it should not
    be discovered via the issuer
    :param token_url: The token endpoint, if it should not be discovered via the issuer
    :param client_secret: The client secret, only needed for confidential clients
    :param scope: The scope(s) to request, space-separated (e.g. ``offline_access`` for
    a refresh token that outlives the single sign-on session)
    :param token_cache: An optional path to a file where the tokens are stored (created
    with owner-only permissions on POSIX systems), so that new runs of a script can
    reuse the previous sign-in instead of opening the browser again. Treat this file
    like a credential.
    :param prompt_callback: How to show the sign-in instructions to the user; defaults
    to :func:`print_device_prompt`, which prints them to stderr
    :param allow_reauthentication: Whether a new interactive sign-in may be started when
    the session has fully expired (default). Set to False for unattended jobs, which
    then fail with :class:`DeviceCodeAuthError` instead of waiting for a browser
    approval that nobody will give.
    :param http_timeout: The timeout in seconds for the requests to the identity
    provider
    :param token_session: The session to use for the requests to the identity provider;
    a new one is created if not given. Do not pass a session that itself
    authenticates with this object — the token requests would then recurse into
    this class and deadlock.
    """

    def __init__(
        self,
        client_id: str,
        issuer_url: str | None = None,
        device_authorization_url: str | None = None,
        token_url: str | None = None,
        client_secret: str | None = None,
        scope: str | None = None,
        token_cache: str | pathlib.Path | None = None,
        prompt_callback: Callable[[str, str, str | None], None] | None = None,
        allow_reauthentication: bool = True,
        http_timeout: float = 30.0,
        token_session: requests.Session | None = None,
    ) -> None:
        if issuer_url is None and (
            device_authorization_url is None or token_url is None
        ):
            raise ValueError(
                "DeviceCodeAuth needs either an issuer_url (for OpenID Connect "
                "discovery) or both a device_authorization_url and a token_url."
            )
        self.client_id = client_id
        self._client_secret = client_secret
        self.scope = scope
        self.issuer_url = issuer_url.rstrip("/") if issuer_url is not None else None
        self.device_authorization_url = device_authorization_url
        self.token_url = token_url
        self.token_cache = (
            pathlib.Path(token_cache).expanduser() if token_cache is not None else None
        )
        self._prompt_callback = (
            prompt_callback if prompt_callback is not None else print_device_prompt
        )
        self._allow_reauthentication = allow_reauthentication
        self._http_timeout = http_timeout
        if token_session is None:
            self._token_session = requests.Session()
        else:
            self._token_session = token_session
        self._lock = threading.Lock()
        self.token: str | None = None
        self._refresh_token: str | None = None
        self._obtained_at = 0.0
        self._expires_in: float | None = None
        if self.device_authorization_url is None or self.token_url is None:
            self._discover_endpoints()
        with self._lock:
            self._load_cache()
        # The initial sign-in may always be interactive; allow_reauthentication only
        # governs what happens once an established session expires later on.
        self._ensure_valid_token(interactive=True)

    # ------------------------------------------------------------------ #
    # requests.auth.AuthBase interface
    # ------------------------------------------------------------------ #

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        """
        Set the Authorization header of the current request, refreshing the token
        beforehand if it is about to expire, and register the retry-on-unauthorized
        hook on the request.

        :param r: The prepared request that should be sent
        :return: The prepared request
        """
        self._ensure_valid_token()
        r.headers["Authorization"] = f"Bearer {self.token}"
        r.register_hook("response", self._handle_unauthorized)
        return r

    def is_refresh_required(self) -> bool:
        """
        Compute whether the access token should be refreshed. Like
        :class:`~fhir_pyrate.util.token_auth.TokenAuth`, the token is refreshed as soon
        as 75% of its lifetime (the ``expires_in`` of the token response) has passed. If
        the server did not communicate a lifetime, the token is only refreshed once a
        request comes back as unauthorized.

        :return: Whether the token is about to expire and should thus be refreshed
        """
        if self.token is None:
            return True
        if self._expires_in is None:
            return False
        elapsed = now_utc().timestamp() - self._obtained_at
        return elapsed >= 0.75 * self._expires_in

    # ------------------------------------------------------------------ #
    # Pickling (multiprocessing support)
    # ------------------------------------------------------------------ #

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Locks cannot be pickled; each process gets its own.
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _discover_endpoints(self) -> None:
        """
        Fill the device authorization and token endpoints from the issuer's OpenID
        Connect discovery document, keeping explicitly given endpoints as they are.
        """
        assert self.issuer_url is not None
        response = self._token_session.get(
            f"{self.issuer_url}/.well-known/openid-configuration",
            timeout=self._http_timeout,
        )
        response.raise_for_status()
        configuration = _json_body(response)
        if self.token_url is None:
            self.token_url = configuration.get("token_endpoint")
        if self.device_authorization_url is None:
            self.device_authorization_url = configuration.get(
                "device_authorization_endpoint"
            )
        if self.token_url is None or self.device_authorization_url is None:
            raise DeviceCodeAuthError(
                f"The OpenID Connect configuration of {self.issuer_url} does not "
                "advertise a token endpoint and a device authorization endpoint, so "
                "the Device Authorization Grant cannot be used. If the server "
                "supports it under non-standard URLs, pass device_authorization_url "
                "and token_url explicitly."
            )

    def _client_data(self) -> dict[str, str]:
        """Build the client identification parameters sent to the identity provider."""
        data = {"client_id": self.client_id}
        if self._client_secret is not None:
            data["client_secret"] = self._client_secret
        return data

    def _ensure_valid_token(
        self, force_refresh: bool = False, interactive: bool | None = None
    ) -> None:
        """
        Make sure that a usable access token is present: keep the current one if it is
        still fresh, otherwise refresh it, otherwise run a new device flow (if
        interactive re-authentication is allowed).

        :param force_refresh: Discard the current access token even if it does not look
        expired (used when the server has just rejected it)
        :param interactive: Whether a new device flow may be started; defaults to the
        ``allow_reauthentication`` setting
        """
        if interactive is None:
            interactive = self._allow_reauthentication
        with self._lock:
            if (
                self.token is not None
                and not force_refresh
                and not self.is_refresh_required()
            ):
                return
            if self._try_refresh():
                return
            if not interactive:
                raise DeviceCodeAuthError(
                    "The access token has expired and could not be refreshed, and "
                    "interactive re-authentication is disabled "
                    "(allow_reauthentication=False). Sign in again manually."
                )
            self._run_device_flow()

    def _try_refresh(self) -> bool:
        """
        Try to obtain a new access token with the refresh token. Must be called with
        the lock held.

        :return: Whether a new access token was obtained
        """
        if self._refresh_token is None:
            return False
        assert self.token_url is not None
        response = self._token_session.post(
            self.token_url,
            data={
                **self._client_data(),
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
            },
            timeout=self._http_timeout,
        )
        if response.ok:
            self._store_tokens(_json_body(response))
            return True
        logger.info(
            "The refresh token was not accepted (HTTP %s), a new sign-in is required.",
            response.status_code,
        )
        self._refresh_token = None
        return False

    def _run_device_flow(self) -> None:
        """
        Run the full Device Authorization Grant: request a device code, show the
        verification URL to the user, and poll the token endpoint until the sign-in
        has been approved. Must be called with the lock held.
        """
        assert self.device_authorization_url is not None
        assert self.token_url is not None
        data = self._client_data()
        if self.scope is not None:
            data["scope"] = self.scope
        response = self._token_session.post(
            self.device_authorization_url, data=data, timeout=self._http_timeout
        )
        response.raise_for_status()
        flow = _json_body(response)
        try:
            device_code = flow["device_code"]
            verification_uri = flow["verification_uri"]
            user_code = flow["user_code"]
        except KeyError as e:
            raise DeviceCodeAuthError(
                f"The device authorization response is missing the {e} field."
            ) from e
        self._prompt_callback(
            verification_uri, user_code, flow.get("verification_uri_complete")
        )
        interval = float(flow.get("interval", 5))
        deadline = time.monotonic() + float(flow.get("expires_in", 600))
        while True:
            if time.monotonic() >= deadline:
                raise DeviceCodeAuthError(
                    "The sign-in was not approved before the device code expired. "
                    "Run your script again to start a new sign-in."
                )
            time.sleep(interval)
            response = self._token_session.post(
                self.token_url,
                data={
                    **self._client_data(),
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                },
                timeout=self._http_timeout,
            )
            if response.ok:
                self._store_tokens(_json_body(response))
                return
            error = _json_body(response).get("error", "")
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                # RFC 8628: increase the polling interval by 5 seconds.
                interval += 5
                continue
            if error == "expired_token":
                raise DeviceCodeAuthError(
                    "The sign-in was not approved before the device code expired. "
                    "Run your script again to start a new sign-in."
                )
            if error == "access_denied":
                raise DeviceCodeAuthError("The sign-in request was denied.")
            response.raise_for_status()
            raise DeviceCodeAuthError(
                f"Unexpected response from the token endpoint: {error or response.text}"
            )

    def _store_tokens(self, payload: dict[str, Any]) -> None:
        """
        Take over the tokens of a successful token response and persist them to the
        token cache (if one is configured). Must be called with the lock held.

        :param payload: The JSON body of the token response
        """
        if "access_token" not in payload:
            raise DeviceCodeAuthError(
                "The token response does not contain an access token."
            )
        self.token = payload["access_token"]
        # A token response may omit the refresh token, in which case the old one
        # stays valid.
        self._refresh_token = payload.get("refresh_token", self._refresh_token)
        expires_in = payload.get("expires_in")
        self._expires_in = float(expires_in) if expires_in is not None else None
        self._obtained_at = now_utc().timestamp()
        self._write_cache()

    def _load_cache(self) -> None:
        """
        Load previously stored tokens from the token cache, ignoring caches that are
        unreadable or belong to a different client or server. Must be called with the
        lock held.
        """
        if self.token_cache is None or not self.token_cache.exists():
            return
        try:
            payload = json.loads(self.token_cache.read_text())
        except (OSError, ValueError):
            logger.debug(
                "The token cache at %s could not be read, ignoring it.",
                self.token_cache,
            )
            return
        if (
            payload.get("client_id") != self.client_id
            or payload.get("token_url") != self.token_url
        ):
            logger.debug(
                "The token cache at %s belongs to a different client or server, "
                "ignoring it.",
                self.token_cache,
            )
            return
        self.token = payload.get("access_token")
        self._refresh_token = payload.get("refresh_token")
        self._obtained_at = float(payload.get("obtained_at", 0.0))
        expires_in = payload.get("expires_in")
        self._expires_in = float(expires_in) if expires_in is not None else None

    def _write_cache(self) -> None:
        """
        Persist the current tokens to the token cache with owner-only permissions
        (on POSIX systems). Must be called with the lock held.
        """
        if self.token_cache is None:
            return
        payload = {
            "client_id": self.client_id,
            "token_url": self.token_url,
            "access_token": self.token,
            "refresh_token": self._refresh_token,
            "obtained_at": self._obtained_at,
            "expires_in": self._expires_in,
        }
        self.token_cache.parent.mkdir(parents=True, exist_ok=True)
        # Write to a per-process temporary file with owner-only permissions from the
        # start (no window in which it is readable by others) and move it into place
        # atomically, so concurrent processes can never see a torn file.
        temporary_path = self.token_cache.parent / (
            f"{self.token_cache.name}.{os.getpid()}.tmp"
        )
        fd = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w") as file:
                json.dump(payload, file)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        os.replace(temporary_path, self.token_cache)

    def _redirects_to_identity_provider(self, response: requests.Response) -> bool:
        """
        Compute whether a response is a redirect to the identity provider. Proxies in
        front of a server (e.g. oauth2-proxy) often answer an expired or missing token
        with a redirect to the sign-in page instead of a plain unauthorized status;
        following it would hand the caller an HTML login page instead of data.

        :param response: The received response
        :return: Whether the response redirects to the host of the identity provider
        """
        if response.status_code not in (301, 302, 303, 307, 308):
            return False
        target = urllib.parse.urlparse(response.headers.get("Location", "")).netloc
        if not target:
            return False
        reference = self.issuer_url if self.issuer_url is not None else self.token_url
        assert reference is not None
        return target == urllib.parse.urlparse(reference).netloc

    def _handle_unauthorized(
        self, response: requests.Response, **kwargs: Any
    ) -> requests.Response:
        """
        Retry a request exactly once with a freshly refreshed token if it came back as
        unauthorized (e.g. because the token expired mid-flight or was revoked) or as a
        redirect to the identity provider's sign-in page. Response hooks run before
        ``requests`` resolves redirects, so the redirect case can be intercepted here.
        This hook is registered per request, so it also works when this auth object is
        attached to a different session than the one it was created for.

        :param response: The received response
        :param kwargs: The send-keyword-arguments of the original request
        :return: The response of the retried request, or the original response
        """
        if (
            response.status_code != requests.codes.unauthorized
            and not self._redirects_to_identity_provider(response)
        ):
            return response
        if getattr(response.request, "device_code_auth_retried", False):
            return response
        if "Authorization" not in response.request.headers:
            # requests strips the Authorization header on cross-host redirects to
            # avoid leaking credentials; then this response was not a rejection of
            # our token, and re-attaching a fresh token here would leak it to a
            # foreign host.
            return response
        connection = getattr(response, "connection", None)
        if connection is None:
            return response
        prepared = response.request.copy()
        # The stubs declare the body as bytes | str | None, but requests also allows
        # file-like objects and generators — exactly the cases that matter here.
        body: Any = prepared.body
        if body is not None and not isinstance(body, (str, bytes)):
            # A file-like or generator body has already been consumed by the first
            # attempt; retrying is only safe if it can be rewound (as requests
            # itself does when following redirects).
            if getattr(prepared, "_body_position", None) is None:
                return response
            try:
                requests.utils.rewind_body(prepared)
            except requests.exceptions.UnrewindableBodyError:
                return response
        # Drain the response so that the connection can be reused.
        response.content  # noqa: B018
        response.close()
        self._ensure_valid_token(force_refresh=True)
        prepared.headers["Authorization"] = f"Bearer {self.token}"
        setattr(prepared, "device_code_auth_retried", True)  # noqa: B010
        retried: requests.Response = connection.send(prepared, **kwargs)
        retried.history.append(response)
        retried.request = prepared
        return retried
