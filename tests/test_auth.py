import unittest
from datetime import timedelta
from typing import Optional, cast
from unittest import mock

import jwt
import requests
from requests.auth import HTTPBasicAuth

from fhir_pyrate import Ahoy
from fhir_pyrate.util import now_utc
from fhir_pyrate.util.token_auth import TokenAuth

AUTH_URL = "https://example.org/auth"
REFRESH_URL = "https://example.org/refresh"
SERVER_TOKEN = "token-from-server"
PRE_EXISTING_TOKEN = "pre-existing-token"


def _fake_response(text: str = "", status_code: int = 200) -> mock.Mock:
    """Build a minimal fake ``requests.Response`` for the token endpoints."""
    response = mock.Mock(spec=requests.Response)
    response.text = text
    response.status_code = status_code

    def raise_for_status() -> None:
        if status_code >= 400:
            raise requests.HTTPError(f"status {status_code}")

    response.raise_for_status.side_effect = raise_for_status
    return response


def _make_jwt(seconds_to_live: int = 3600, age_seconds: int = 0) -> str:
    """Create a JWT whose lifetime is ``seconds_to_live`` and that was issued
    ``age_seconds`` ago, so refresh logic can be exercised deterministically."""
    issued = now_utc().timestamp() - age_seconds
    return str(
        jwt.encode(
            {"iat": int(issued), "exp": int(issued + seconds_to_live)},
            "test-secret-key-of-sufficient-length-32b",
            algorithm="HS256",
        )
    )


def _bearer_for(auth: requests.auth.AuthBase) -> Optional[str]:
    """Run an auth handler against a dummy request and return its Authorization header."""
    request = requests.Request("GET", "https://example.org/fhir/Patient").prepare()
    auth(request)
    return request.headers.get("Authorization")


class AuthTest(unittest.TestCase):
    """Exercises every authentication method offered by :class:`Ahoy` / :class:`TokenAuth`
    without touching the network."""

    # ------------------------------------------------------------------ #
    # token auth_type
    # ------------------------------------------------------------------ #
    def testTokenWithPasswordMethod(self) -> None:
        with (
            mock.patch.object(
                requests.Session, "get", return_value=_fake_response(SERVER_TOKEN)
            ) as get,
            mock.patch("fhir_pyrate.ahoy.getpass.getpass", return_value="secret"),
        ):
            ahoy = Ahoy(
                auth_type="token",
                auth_method="password",
                auth_url=AUTH_URL,
                username="alice",
            )
        auth = ahoy.session.auth
        assert isinstance(auth, TokenAuth)
        self.assertEqual(auth.token, SERVER_TOKEN)
        self.assertEqual(_bearer_for(auth), f"Bearer {SERVER_TOKEN}")
        # The auth endpoint was queried with BasicAuth credentials.
        get.assert_called_with(AUTH_URL, auth=("alice", "secret"))

    def testTokenWithEnvMethod(self) -> None:
        env = {"FHIR_USER": "bob", "FHIR_PASSWORD": "hunter2"}
        with (
            mock.patch.dict("os.environ", env, clear=False),
            mock.patch.object(
                requests.Session, "get", return_value=_fake_response(SERVER_TOKEN)
            ) as get,
        ):
            ahoy = Ahoy(auth_type="token", auth_method="env", auth_url=AUTH_URL)
        auth = ahoy.session.auth
        self.assertIsInstance(auth, TokenAuth)
        auth = cast(TokenAuth, auth)
        self.assertEqual(auth.token, SERVER_TOKEN)
        get.assert_called_with(AUTH_URL, auth=("bob", "hunter2"))

    def testTokenWithKeyringMethodNotImplemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            Ahoy(auth_type="token", auth_method="keyring", auth_url=AUTH_URL)

    def testTokenWithoutAuthUrlFails(self) -> None:
        # password method, token type, but no auth_url -> assertion error
        with mock.patch("fhir_pyrate.ahoy.getpass.getpass", return_value="secret"):
            with self.assertRaises(AssertionError):
                Ahoy(
                    auth_type="token",
                    auth_method="password",
                    username="alice",
                    auth_url=None,
                )

    # ------------------------------------------------------------------ #
    # BasicAuth auth_type
    # ------------------------------------------------------------------ #
    def testBasicAuthWithPasswordMethod(self) -> None:
        with mock.patch("fhir_pyrate.ahoy.getpass.getpass", return_value="secret"):
            ahoy = Ahoy(
                auth_type="BasicAuth",
                auth_method="password",
                username="alice",
            )
        auth = ahoy.session.auth
        assert isinstance(auth, HTTPBasicAuth)
        self.assertEqual(auth.username, "alice")
        self.assertEqual(auth.password, "secret")

    def testBasicAuthWithEnvMethod(self) -> None:
        env = {"FHIR_USER": "bob", "FHIR_PASSWORD": "hunter2"}
        with mock.patch.dict("os.environ", env, clear=False):
            ahoy = Ahoy(auth_type="BasicAuth", auth_method="env")
        auth = ahoy.session.auth
        assert isinstance(auth, HTTPBasicAuth)
        self.assertEqual(auth.username, "bob")
        self.assertEqual(auth.password, "hunter2")

    # ------------------------------------------------------------------ #
    # Pre-existing token
    # ------------------------------------------------------------------ #
    def testPreExistingTokenNoLogin(self) -> None:
        with mock.patch.object(requests.Session, "get") as get:
            ahoy = Ahoy(token=PRE_EXISTING_TOKEN)
        auth = ahoy.session.auth
        assert isinstance(auth, TokenAuth)
        self.assertEqual(auth.token, PRE_EXISTING_TOKEN)
        self.assertEqual(_bearer_for(auth), f"Bearer {PRE_EXISTING_TOKEN}")
        # No login should have been attempted.
        get.assert_not_called()

    def testPreExistingTokenWithoutAuthMethod(self) -> None:
        # auth_method explicitly None should still authenticate via the token.
        ahoy = Ahoy(token=PRE_EXISTING_TOKEN, auth_method=None)
        auth = ahoy.session.auth
        assert isinstance(auth, TokenAuth)
        self.assertEqual(auth.token, PRE_EXISTING_TOKEN)

    def testPreExistingTokenRejectedForBasicAuth(self) -> None:
        with self.assertRaises(ValueError):
            Ahoy(auth_type="BasicAuth", token=PRE_EXISTING_TOKEN)

    # ------------------------------------------------------------------ #
    # Invalid configurations
    # ------------------------------------------------------------------ #
    def testUndefinedAuthMethod(self) -> None:
        with self.assertRaises(ValueError):
            Ahoy(auth_type="token", auth_method="does-not-exist", auth_url=AUTH_URL)

    def testUndefinedAuthType(self) -> None:
        env = {"FHIR_USER": "bob", "FHIR_PASSWORD": "hunter2"}
        with mock.patch.dict("os.environ", env, clear=False):
            with self.assertRaises(ValueError):
                Ahoy(auth_type="does-not-exist", auth_method="env")

    def testTokenAuthRequiresTokenOrAuthUrl(self) -> None:
        with self.assertRaises(ValueError):
            TokenAuth()

    # ------------------------------------------------------------------ #
    # TokenAuth refresh behaviour
    # ------------------------------------------------------------------ #
    def testRefreshNotRequiredForFreshJwt(self) -> None:
        auth = TokenAuth(token=_make_jwt(seconds_to_live=3600, age_seconds=0))
        self.assertFalse(auth.is_refresh_required())

    def testRefreshRequiredForExpiringJwt(self) -> None:
        # 90% of the lifetime has passed -> within the last 25%, refresh needed.
        auth = TokenAuth(token=_make_jwt(seconds_to_live=1000, age_seconds=900))
        self.assertTrue(auth.is_refresh_required())

    def testRefreshNotRequiredForOpaqueTokenWithoutDelta(self) -> None:
        auth = TokenAuth(token="opaque-not-a-jwt")
        self.assertFalse(auth.is_refresh_required())

    def testRefreshRequiredForOpaqueTokenWithExpiredDelta(self) -> None:
        auth = TokenAuth(
            token="opaque-not-a-jwt", token_refresh_delta=timedelta(minutes=5)
        )
        # Pretend the token was obtained ten minutes ago.
        auth.auth_time = now_utc() - timedelta(minutes=10)
        self.assertTrue(auth.is_refresh_required())

    def testRefreshWithExplicitToken(self) -> None:
        auth = TokenAuth(token="old-token")
        auth.refresh_token(token="new-token")
        self.assertEqual(auth.token, "new-token")

    def testRefreshViaRefreshUrl(self) -> None:
        auth = TokenAuth(token="old-token", refresh_url=REFRESH_URL)
        with mock.patch.object(
            auth._token_session, "get", return_value=_fake_response("refreshed-token")
        ) as get:
            auth.refresh_token()
        self.assertEqual(auth.token, "refreshed-token")
        get.assert_called_with(REFRESH_URL)

    def testRefreshViaReAuthentication(self) -> None:
        auth = TokenAuth(token="old-token", auth_url=AUTH_URL)
        with mock.patch.object(
            auth._token_session, "get", return_value=_fake_response("re-authed-token")
        ) as get:
            auth.refresh_token()
        self.assertEqual(auth.token, "re-authed-token")
        get.assert_called_with(AUTH_URL, auth=None)

    def testRefreshWithoutAnyMeansFails(self) -> None:
        # Token-only setup with no auth_url and no refresh_url cannot be refreshed.
        auth = TokenAuth(token="old-token")
        with self.assertRaises(ValueError):
            auth.refresh_token()


if __name__ == "__main__":
    unittest.main()
