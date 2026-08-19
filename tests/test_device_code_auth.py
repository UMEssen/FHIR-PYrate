import io
import json
import pathlib
import pickle
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import requests

from fhir_pyrate import Ahoy
from fhir_pyrate.util import now_utc
from fhir_pyrate.util.device_code_auth import DeviceCodeAuth, DeviceCodeAuthError

CLIENT_ID = "example-cli"
ISSUER_URL = "https://login.example.org/realms/example"
DEVICE_URL = "https://login.example.org/device"
TOKEN_URL = "https://login.example.org/token"

DISCOVERY_RESPONSE = {
    "token_endpoint": TOKEN_URL,
    "device_authorization_endpoint": DEVICE_URL,
}
DEVICE_RESPONSE = {
    "device_code": "device-code-123",
    "user_code": "ABCD-EFGH",
    "verification_uri": "https://login.example.org/device-verification",
    "verification_uri_complete": (
        "https://login.example.org/device-verification?user_code=ABCD-EFGH"
    ),
    "expires_in": 600,
    "interval": 5,
}
TOKEN_RESPONSE = {
    "access_token": "access-token-1",
    "refresh_token": "refresh-token-1",
    "expires_in": 300,
}
PENDING_RESPONSE = {"error": "authorization_pending"}


def _json_response(payload: Dict[str, Any], status_code: int = 200) -> mock.Mock:
    """Build a minimal fake ``requests.Response`` with a JSON body."""
    response = mock.Mock(spec=requests.Response)
    response.status_code = status_code
    response.ok = status_code < 400
    response.json.return_value = payload
    response.text = json.dumps(payload)

    def raise_for_status() -> None:
        if status_code >= 400:
            raise requests.HTTPError(f"status {status_code}")

    response.raise_for_status.side_effect = raise_for_status
    return response


def _bearer_for(auth: requests.auth.AuthBase) -> Optional[str]:
    """Run an auth handler against a dummy request and return its Authorization header."""
    request = requests.Request("GET", "https://example.org/fhir/Patient").prepare()
    auth(request)
    return request.headers.get("Authorization")


def _make_auth(
    post_responses: Optional[List[mock.Mock]] = None, **kwargs: Any
) -> DeviceCodeAuth:
    """Construct a DeviceCodeAuth going through discovery, one pending poll and a
    successful token response, without touching the network."""
    if post_responses is None:
        post_responses = [
            _json_response(DEVICE_RESPONSE),
            _json_response(PENDING_RESPONSE, status_code=400),
            _json_response(TOKEN_RESPONSE),
        ]
    with (
        mock.patch.object(
            requests.Session, "get", return_value=_json_response(DISCOVERY_RESPONSE)
        ),
        mock.patch.object(requests.Session, "post", side_effect=post_responses),
        mock.patch("fhir_pyrate.util.device_code_auth.time.sleep"),
    ):
        return DeviceCodeAuth(
            client_id=CLIENT_ID,
            issuer_url=ISSUER_URL,
            prompt_callback=kwargs.pop("prompt_callback", lambda *args: None),
            **kwargs,
        )


class DeviceCodeAuthTest(unittest.TestCase):
    """Exercises the Device Authorization Grant without touching the network."""

    # ------------------------------------------------------------------ #
    # Construction & discovery
    # ------------------------------------------------------------------ #
    def testNeedsIssuerOrExplicitEndpoints(self) -> None:
        with self.assertRaises(ValueError):
            DeviceCodeAuth(client_id=CLIENT_ID)
        with self.assertRaises(ValueError):
            DeviceCodeAuth(client_id=CLIENT_ID, token_url=TOKEN_URL)

    def testDiscoveryWithoutDeviceEndpointFails(self) -> None:
        with (
            mock.patch.object(
                requests.Session,
                "get",
                return_value=_json_response({"token_endpoint": TOKEN_URL}),
            ),
            self.assertRaises(DeviceCodeAuthError),
        ):
            DeviceCodeAuth(client_id=CLIENT_ID, issuer_url=ISSUER_URL)

    # ------------------------------------------------------------------ #
    # The device flow itself
    # ------------------------------------------------------------------ #
    def testHappyPath(self) -> None:
        prompts: List[Tuple[str, str, Optional[str]]] = []
        auth = _make_auth(
            prompt_callback=lambda uri, code, complete: prompts.append(
                (uri, code, complete)
            )
        )
        self.assertEqual(auth.token, TOKEN_RESPONSE["access_token"])
        self.assertEqual(
            _bearer_for(auth), f"Bearer {TOKEN_RESPONSE['access_token']}"
        )
        self.assertEqual(
            prompts,
            [
                (
                    DEVICE_RESPONSE["verification_uri"],
                    DEVICE_RESPONSE["user_code"],
                    DEVICE_RESPONSE["verification_uri_complete"],
                )
            ],
        )

    def testSlowDownIncreasesPollingInterval(self) -> None:
        post_responses = [
            _json_response(DEVICE_RESPONSE),
            _json_response({"error": "slow_down"}, status_code=400),
            _json_response(PENDING_RESPONSE, status_code=400),
            _json_response(TOKEN_RESPONSE),
        ]
        with (
            mock.patch.object(
                requests.Session, "get", return_value=_json_response(DISCOVERY_RESPONSE)
            ),
            mock.patch.object(requests.Session, "post", side_effect=post_responses),
            mock.patch("fhir_pyrate.util.device_code_auth.time.sleep") as sleep,
        ):
            DeviceCodeAuth(
                client_id=CLIENT_ID,
                issuer_url=ISSUER_URL,
                prompt_callback=lambda *args: None,
            )
        # Base interval of 5, increased by 5 after the slow_down.
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10, 10])

    def testAccessDeniedFails(self) -> None:
        with self.assertRaises(DeviceCodeAuthError):
            _make_auth(
                post_responses=[
                    _json_response(DEVICE_RESPONSE),
                    _json_response({"error": "access_denied"}, status_code=400),
                ]
            )

    def testExpiredDeviceCodeFails(self) -> None:
        with self.assertRaises(DeviceCodeAuthError):
            _make_auth(
                post_responses=[
                    _json_response(DEVICE_RESPONSE),
                    _json_response({"error": "expired_token"}, status_code=400),
                ]
            )

    def testDeadlineExceededFails(self) -> None:
        expired_flow = dict(DEVICE_RESPONSE, expires_in=0)
        with self.assertRaises(DeviceCodeAuthError):
            _make_auth(post_responses=[_json_response(expired_flow)])

    # ------------------------------------------------------------------ #
    # Refresh behaviour
    # ------------------------------------------------------------------ #
    def testRefreshNotRequiredForFreshToken(self) -> None:
        auth = _make_auth()
        self.assertFalse(auth.is_refresh_required())

    def testRefreshRequiredInLastQuarterOfLifetime(self) -> None:
        auth = _make_auth()
        # Pretend the token was obtained 80% of its lifetime ago.
        auth._obtained_at = now_utc().timestamp() - 0.8 * TOKEN_RESPONSE["expires_in"]
        self.assertTrue(auth.is_refresh_required())

    def testProactiveRefreshOnCall(self) -> None:
        auth = _make_auth()
        auth._obtained_at = now_utc().timestamp() - 0.8 * TOKEN_RESPONSE["expires_in"]
        refreshed = dict(TOKEN_RESPONSE, access_token="access-token-2")
        with mock.patch.object(
            requests.Session, "post", return_value=_json_response(refreshed)
        ) as post:
            self.assertEqual(_bearer_for(auth), "Bearer access-token-2")
        self.assertEqual(
            post.call_args.kwargs["data"]["grant_type"], "refresh_token"
        )
        self.assertEqual(
            post.call_args.kwargs["data"]["refresh_token"],
            TOKEN_RESPONSE["refresh_token"],
        )

    def testRejectedRefreshFallsBackToDeviceFlow(self) -> None:
        auth = _make_auth()
        auth._obtained_at = 0.0  # long expired
        second_token = dict(TOKEN_RESPONSE, access_token="access-token-2")
        post_responses = [
            _json_response({"error": "invalid_grant"}, status_code=400),  # refresh
            _json_response(DEVICE_RESPONSE),  # new device flow
            _json_response(second_token),
        ]
        with (
            mock.patch.object(requests.Session, "post", side_effect=post_responses),
            mock.patch("fhir_pyrate.util.device_code_auth.time.sleep"),
        ):
            self.assertEqual(_bearer_for(auth), "Bearer access-token-2")

    def testRejectedRefreshWithoutReauthenticationFails(self) -> None:
        auth = _make_auth(allow_reauthentication=False)
        auth._obtained_at = 0.0  # long expired
        with (
            mock.patch.object(
                requests.Session,
                "post",
                return_value=_json_response(
                    {"error": "invalid_grant"}, status_code=400
                ),
            ),
            self.assertRaises(DeviceCodeAuthError),
        ):
            _bearer_for(auth)

    # ------------------------------------------------------------------ #
    # Token cache
    # ------------------------------------------------------------------ #
    def testTokenCacheIsWrittenWithOwnerOnlyPermissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = pathlib.Path(tmp_dir) / "tokens.json"
            _make_auth(token_cache=cache_file)
            payload = json.loads(cache_file.read_text())
            self.assertEqual(payload["access_token"], TOKEN_RESPONSE["access_token"])
            self.assertEqual(payload["refresh_token"], TOKEN_RESPONSE["refresh_token"])
            self.assertEqual(payload["client_id"], CLIENT_ID)
            self.assertEqual(cache_file.stat().st_mode & 0o777, 0o600)

    def testFreshCachedTokenNeedsNoNetwork(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = pathlib.Path(tmp_dir) / "tokens.json"
            _make_auth(token_cache=cache_file)
            with (
                mock.patch.object(requests.Session, "get") as get,
                mock.patch.object(requests.Session, "post") as post,
            ):
                auth = DeviceCodeAuth(
                    client_id=CLIENT_ID,
                    device_authorization_url=DEVICE_URL,
                    token_url=TOKEN_URL,
                    token_cache=cache_file,
                    prompt_callback=lambda *args: None,
                )
            self.assertEqual(auth.token, TOKEN_RESPONSE["access_token"])
            get.assert_not_called()
            post.assert_not_called()

    def testCacheOfDifferentClientIsIgnored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = pathlib.Path(tmp_dir) / "tokens.json"
            _make_auth(token_cache=cache_file)
            prompts: List[str] = []
            with (
                mock.patch.object(
                    requests.Session,
                    "get",
                    return_value=_json_response(DISCOVERY_RESPONSE),
                ),
                mock.patch.object(
                    requests.Session,
                    "post",
                    side_effect=[
                        _json_response(DEVICE_RESPONSE),
                        _json_response(TOKEN_RESPONSE),
                    ],
                ),
                mock.patch("fhir_pyrate.util.device_code_auth.time.sleep"),
            ):
                DeviceCodeAuth(
                    client_id="another-client",
                    issuer_url=ISSUER_URL,
                    token_cache=cache_file,
                    prompt_callback=lambda *args: prompts.append(args[0]),
                )
            # The cache did not match, so a new device flow (with prompt) was run.
            self.assertEqual(len(prompts), 1)

    # ------------------------------------------------------------------ #
    # Pickling (multiprocessing support)
    # ------------------------------------------------------------------ #
    def testPickleRoundTrip(self) -> None:
        auth = _make_auth(prompt_callback=None)
        restored = pickle.loads(pickle.dumps(auth))
        self.assertEqual(restored.token, auth.token)
        self.assertEqual(
            _bearer_for(restored), f"Bearer {TOKEN_RESPONSE['access_token']}"
        )

    # ------------------------------------------------------------------ #
    # Retry on unauthorized
    # ------------------------------------------------------------------ #
    def testUnauthorizedResponseIsRetriedOnce(self) -> None:
        auth = _make_auth()
        request = requests.Request("GET", "https://example.org/fhir/Patient").prepare()
        auth(request)
        unauthorized = mock.Mock(spec=requests.Response)
        unauthorized.status_code = requests.codes.unauthorized
        unauthorized.request = request
        retried_response = mock.Mock(spec=requests.Response)
        retried_response.status_code = 200
        retried_response.history = []
        unauthorized.connection = mock.Mock()
        unauthorized.connection.send.return_value = retried_response
        refreshed = dict(TOKEN_RESPONSE, access_token="access-token-2")
        with mock.patch.object(
            requests.Session, "post", return_value=_json_response(refreshed)
        ):
            result = auth._handle_unauthorized(unauthorized)
        self.assertIs(result, retried_response)
        self.assertEqual(result.history, [unauthorized])
        retried_request = unauthorized.connection.send.call_args.args[0]
        self.assertEqual(
            retried_request.headers["Authorization"], "Bearer access-token-2"
        )
        # The retried request is marked, so a second 401 would not loop.
        second = mock.Mock(spec=requests.Response)
        second.status_code = requests.codes.unauthorized
        second.request = retried_request
        self.assertIs(auth._handle_unauthorized(second), second)

    def testRedirectToIdentityProviderIsRetried(self) -> None:
        auth = _make_auth()
        request = requests.Request("GET", "https://example.org/fhir/Patient").prepare()
        auth(request)
        redirect = mock.Mock(spec=requests.Response)
        redirect.status_code = 302
        redirect.headers = {
            "Location": (
                "https://login.example.org/realms/example/protocol/"
                "openid-connect/auth?client_id=proxy"
            )
        }
        redirect.request = request
        retried_response = mock.Mock(spec=requests.Response)
        retried_response.status_code = 200
        retried_response.history = []
        redirect.connection = mock.Mock()
        redirect.connection.send.return_value = retried_response
        refreshed = dict(TOKEN_RESPONSE, access_token="access-token-2")
        with mock.patch.object(
            requests.Session, "post", return_value=_json_response(refreshed)
        ):
            result = auth._handle_unauthorized(redirect)
        self.assertIs(result, retried_response)

    def testRedirectToOtherHostIsNotIntercepted(self) -> None:
        auth = _make_auth()
        redirect = mock.Mock(spec=requests.Response)
        redirect.status_code = 302
        redirect.headers = {"Location": "https://elsewhere.example.org/moved"}
        self.assertIs(auth._handle_unauthorized(redirect), redirect)

    def testStrippedAuthorizationHeaderIsNotRetried(self) -> None:
        # requests removes the Authorization header on cross-host redirects; a 401
        # for such a request must NOT be answered with a fresh token, which would
        # leak it to the foreign host.
        auth = _make_auth()
        request = requests.Request("GET", "https://other-host.example.org/x").prepare()
        self.assertNotIn("Authorization", request.headers)
        unauthorized = mock.Mock(spec=requests.Response)
        unauthorized.status_code = requests.codes.unauthorized
        unauthorized.request = request
        unauthorized.connection = mock.Mock()
        with mock.patch.object(requests.Session, "post") as post:
            self.assertIs(auth._handle_unauthorized(unauthorized), unauthorized)
        post.assert_not_called()
        unauthorized.connection.send.assert_not_called()

    def testConsumedStreamBodyIsNotRetried(self) -> None:
        # A generator body cannot be rewound; retrying would send a truncated body.
        auth = _make_auth()
        request = requests.Request(
            "POST",
            "https://example.org/fhir/Binary",
            data=(chunk for chunk in [b"chunk"]),
        ).prepare()
        auth(request)
        unauthorized = mock.Mock(spec=requests.Response)
        unauthorized.status_code = requests.codes.unauthorized
        unauthorized.request = request
        unauthorized.connection = mock.Mock()
        self.assertIs(auth._handle_unauthorized(unauthorized), unauthorized)
        unauthorized.connection.send.assert_not_called()

    def testRewindableBodyIsRewoundOnRetry(self) -> None:
        auth = _make_auth()
        body = io.BytesIO(b"dicom-payload")
        request = requests.Request(
            "POST", "https://example.org/studies", data=body
        ).prepare()
        auth(request)
        body.read()  # The first (failed) attempt consumed the stream.
        unauthorized = mock.Mock(spec=requests.Response)
        unauthorized.status_code = requests.codes.unauthorized
        unauthorized.request = request
        retried_response = mock.Mock(spec=requests.Response)
        retried_response.status_code = 200
        retried_response.history = []
        unauthorized.connection = mock.Mock()
        unauthorized.connection.send.return_value = retried_response
        with mock.patch.object(
            requests.Session, "post", return_value=_json_response(TOKEN_RESPONSE)
        ):
            result = auth._handle_unauthorized(unauthorized)
        self.assertIs(result, retried_response)
        # The body was rewound, so the retry sends the complete payload again.
        self.assertEqual(body.tell(), 0)

    # ------------------------------------------------------------------ #
    # Ahoy wiring
    # ------------------------------------------------------------------ #
    def testAhoyDeviceCode(self) -> None:
        with (
            mock.patch.object(
                requests.Session, "get", return_value=_json_response(DISCOVERY_RESPONSE)
            ),
            mock.patch.object(
                requests.Session,
                "post",
                side_effect=[
                    _json_response(DEVICE_RESPONSE),
                    _json_response(TOKEN_RESPONSE),
                ],
            ),
            mock.patch("fhir_pyrate.util.device_code_auth.time.sleep"),
            mock.patch(
                "fhir_pyrate.util.device_code_auth.print_device_prompt"
            ) as prompt,
        ):
            ahoy = Ahoy(
                auth_type="device_code",
                auth_url=ISSUER_URL,
                auth_method=None,
                client_id=CLIENT_ID,
            )
        auth = ahoy.session.auth
        assert isinstance(auth, DeviceCodeAuth)
        self.assertEqual(auth.token, TOKEN_RESPONSE["access_token"])
        prompt.assert_called_once()

    def testAhoyDeviceCodeNeedsClientId(self) -> None:
        with self.assertRaises(ValueError):
            Ahoy(auth_type="device_code", auth_url=ISSUER_URL)

    def testAhoyDeviceCodeNeedsIssuer(self) -> None:
        with self.assertRaises(ValueError):
            Ahoy(auth_type="device_code", client_id=CLIENT_ID)


class _FakeIdentityProviderHandler(BaseHTTPRequestHandler):
    """A minimal OAuth 2.0 identity provider + protected resource, so the whole flow
    can be tested over real sockets with a real ``requests.Session``."""

    server: "_FakeIdentityProviderServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        state = self.server.state
        if self.path == "/.well-known/openid-configuration":
            base = f"http://127.0.0.1:{self.server.server_port}"
            self._send_json(
                {
                    "token_endpoint": f"{base}/token",
                    "device_authorization_endpoint": f"{base}/device",
                }
            )
        elif self.path == "/resource":
            header = self.headers.get("Authorization", "")
            if header == f"Bearer {state['current_access_token']}":
                self._send_json({"resourceType": "Bundle"})
            elif state["challenge"] == "redirect":
                # Proxy-style challenge: redirect to the identity provider's
                # sign-in page instead of a plain 401 (like oauth2-proxy).
                base = f"http://127.0.0.1:{self.server.server_port}"
                self.send_response(302)
                self.send_header("Location", f"{base}/sign-in")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self._send_json({"error": "invalid_token"}, status=401)
        else:
            self._send_json({"error": "not_found"}, status=404)

    def _issue_tokens(self) -> None:
        state = self.server.state
        state["counter"] += 1
        state["current_access_token"] = f"access-{state['counter']}"
        self._send_json(
            {
                "access_token": state["current_access_token"],
                "refresh_token": f"refresh-{state['counter']}",
                "expires_in": 300,
            }
        )

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        form = urllib.parse.parse_qs(self.rfile.read(length).decode())
        state = self.server.state
        if self.path == "/device":
            state["polls"] = 0
            self._send_json(
                {
                    "device_code": "integration-device-code",
                    "user_code": "WXYZ-1234",
                    "verification_uri": "http://127.0.0.1/device-verification",
                    "expires_in": 60,
                    # Poll without sleeping to keep the test fast.
                    "interval": 0,
                }
            )
        elif self.path == "/token":
            grant_type = form.get("grant_type", [""])[0]
            if grant_type == "urn:ietf:params:oauth:grant-type:device_code":
                state["polls"] += 1
                if state["polls"] < 2:
                    self._send_json({"error": "authorization_pending"}, status=400)
                else:
                    self._issue_tokens()
            elif grant_type == "refresh_token":
                if form.get("refresh_token") == [f"refresh-{state['counter']}"]:
                    self._issue_tokens()
                else:
                    self._send_json({"error": "invalid_grant"}, status=400)
            else:
                self._send_json({"error": "unsupported_grant_type"}, status=400)
        else:
            self._send_json({"error": "not_found"}, status=404)


class _FakeIdentityProviderServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _FakeIdentityProviderHandler)
        self.state: Dict[str, Any] = {
            "counter": 0,
            "polls": 0,
            "current_access_token": None,
            "challenge": "401",
        }


class DeviceCodeAuthIntegrationTest(unittest.TestCase):
    """Runs the complete flow (discovery, device flow, refresh, retry on 401) against
    an in-process identity provider over real sockets."""

    def setUp(self) -> None:
        self.server = _FakeIdentityProviderServer()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.issuer = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def testFullFlowAgainstLocalIdentityProvider(self) -> None:
        prompts: List[str] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = pathlib.Path(tmp_dir) / "tokens.json"
            auth = DeviceCodeAuth(
                client_id=CLIENT_ID,
                issuer_url=self.issuer,
                token_cache=cache_file,
                prompt_callback=lambda uri, code, complete: prompts.append(code),
            )
            self.assertEqual(prompts, ["WXYZ-1234"])
            with requests.Session() as session:
                session.auth = auth
                response = session.get(f"{self.issuer}/resource")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"resourceType": "Bundle"})

                # Simulate a token revocation on the server: the next request is
                # answered with 401, and the auth transparently refreshes + retries.
                self.server.state["current_access_token"] = "revoked"
                response = session.get(f"{self.issuer}/resource")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    [r.status_code for r in response.history],
                    [requests.codes.unauthorized],
                )

            # A second instance reuses the cached tokens: no new device flow.
            second_prompts: List[str] = []
            second = DeviceCodeAuth(
                client_id=CLIENT_ID,
                issuer_url=self.issuer,
                token_cache=cache_file,
                prompt_callback=lambda uri, code, complete: second_prompts.append(code),
            )
            self.assertEqual(second_prompts, [])
            with requests.Session() as session:
                session.auth = second
                response = session.get(f"{self.issuer}/resource")
                self.assertEqual(response.status_code, 200)

    def testProxyStyleRedirectChallengeIsRetried(self) -> None:
        auth = DeviceCodeAuth(
            client_id=CLIENT_ID,
            issuer_url=self.issuer,
            prompt_callback=lambda *args: None,
        )
        # The server now answers bad tokens with a redirect to its sign-in page
        # (like oauth2-proxy) and considers the current token revoked.
        self.server.state["challenge"] = "redirect"
        self.server.state["current_access_token"] = "revoked"
        with requests.Session() as session:
            session.auth = auth
            response = session.get(f"{self.issuer}/resource")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"resourceType": "Bundle"})
            self.assertEqual([r.status_code for r in response.history], [302])


if __name__ == "__main__":
    unittest.main()
