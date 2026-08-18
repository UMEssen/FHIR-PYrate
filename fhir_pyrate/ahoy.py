import getpass
import logging
import os
import pathlib
from datetime import timedelta
from types import TracebackType

import requests
from requests.auth import HTTPBasicAuth

from fhir_pyrate.util.device_code_auth import DeviceCodeAuth
from fhir_pyrate.util.token_auth import TokenAuth

logger = logging.getLogger(__name__)


class Ahoy:
    """
    Simple authentication class that supports token authentication, BasicAuth and the
    OAuth 2.0 Device Authorization Grant.

    :param auth_url: The URL to use for authentication; for the "device_code"
    authentication type this is the OpenID Connect issuer URL (e.g.
    https://keycloak.example.com/realms/example-realm)
    :param auth_type: The kind of authentication, for now "token", "BasicAuth" and
    "device_code" are supported.
    :param refresh_url:  The URL to use to refresh the token
    :param username: The username to use for the authentication (for the password authentication
    method)
    :param auth_method: The options are [password, env, keyring]:
    password will use the given username as username and ask to input a password;
    env will use the environment variables FHIR_USER and FHIR_PASSWORD;
    keyring will use a keyring [NOT IMPLEMENTED YET].
    This parameter is ignored by the "device_code" authentication type, which never
    needs a password.
    :param token: A pre-existing token to use for authentication. If this is given, no login is
    performed and the token is used as-is, so the username/password/auth_method variables do not
    need to be specified. An auth_url and/or refresh_url are then only needed if the token should
    be refreshed once it expires.
    :param max_login_attempts: The maximum number of logins that can be performed
    :param token_refresh_delta: Either a timedelta object that tells us how often the token
    should be refreshed, or a number of minutes; this does not need to be specified for JWT tokens
    that contain the expiry date
    :param session: The session that can be used for the authentication. This is particularly
    useful if you have some particular requirements for your authentication (e.g. you need to
    support for cusum self-signed certificates).
    :param client_id: The OAuth client ID (only for the "device_code" authentication
    type)
    :param client_secret: The OAuth client secret, only needed for confidential clients
    (only for the "device_code" authentication type)
    :param scope: The OAuth scope(s) to request, space-separated, e.g. "offline_access"
    (only for the "device_code" authentication type)
    :param token_cache: An optional path to a file where the tokens of the device flow
    are stored, so that new runs can reuse the previous sign-in instead of opening the
    browser again; treat this file like a credential (only for the "device_code"
    authentication type)
    """

    def __init__(
        self,
        auth_url: str | None = None,
        auth_type: str | None = "token",
        refresh_url: str | None = None,
        username: str | None = None,
        auth_method: str | None = "password",
        token: str | None = None,
        max_login_attempts: int = 5,
        token_refresh_delta: int | timedelta | None = None,
        session: requests.Session | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        scope: str | None = None,
        token_cache: str | pathlib.Path | None = None,
    ) -> None:
        self.auth_type = auth_type
        self.auth_method = auth_method
        self.auth_url = auth_url
        self.refresh_url = refresh_url
        self.username = username
        self._user_env_name = "FHIR_USER"
        self._pass_env_name = "FHIR_PASSWORD"  # noqa: S105
        self.token = token
        if session is None:
            self.session = requests.Session()
        else:
            self.session = session
        self.max_login_attempts = max_login_attempts
        self.token_refresh_delta = token_refresh_delta
        self.client_id = client_id
        self._client_secret = client_secret
        self.scope = scope
        self.token_cache = token_cache
        if self.token is not None or (
            self.auth_type is not None
            and (
                self.auth_method is not None
                # The device flow needs no username/password, so it must not depend
                # on an auth_method being set.
                or self.auth_type.lower() == "device_code"
            )
        ):
            self._authenticate()

    def __enter__(self) -> "Ahoy":
        return self

    def close(self) -> None:
        self.session.close()

    def __exit__(
        self,
        exctype: type[BaseException] | None,
        excinst: BaseException | None,
        exctb: TracebackType | None,
    ) -> None:
        self.close()

    def change_environment_variable_name(
        self, user_env: str | None = None, pass_env: str | None = None
    ) -> None:
        """
        Change the name of the variables used to retrieve username and password.

        :param user_env: The future name of the username variable
        :param pass_env: The future name of the password variable
        :return: None
        """
        if user_env is not None:
            self._user_env_name = user_env
        if pass_env is not None:
            self._pass_env_name = pass_env

    def _authenticate(self) -> None:
        """
        Authenticate the user in the current session with a token or with BasicAuth.
        """
        assert self.auth_type is not None
        if self.token is not None:
            if self.auth_type.lower() != "token":
                raise ValueError(
                    "A pre-existing token can only be used with the 'token' "
                    f"authentication type, but {self.auth_type} was given."
                )
            self.session.auth = TokenAuth(
                auth_url=self.auth_url,
                refresh_url=self.refresh_url,
                session=self.session,
                max_login_attempts=self.max_login_attempts,
                token_refresh_delta=self.token_refresh_delta,
                token=self.token,
            )
            return
        if self.auth_type.lower() == "device_code":
            if self.auth_url is None:
                raise ValueError(
                    "The device_code authentication type needs an auth_url that "
                    "points to the OpenID Connect issuer, e.g. "
                    "https://keycloak.example.com/realms/example-realm."
                )
            if self.client_id is None:
                raise ValueError(
                    "The device_code authentication type needs a client_id."
                )
            self.session.auth = DeviceCodeAuth(
                client_id=self.client_id,
                issuer_url=self.auth_url,
                client_secret=self._client_secret,
                scope=self.scope,
                token_cache=self.token_cache,
            )
            return
        assert self.auth_method is not None
        if self.auth_method.lower() == "password":
            assert self.username is not None, (
                "When using the password authentication method, "
                "a username should be given as input."
            )
            username = self.username
            password = getpass.getpass()
        elif self.auth_method.lower() == "env":
            username = os.environ[self._user_env_name]
            password = os.environ[self._pass_env_name]
        elif self.auth_method.lower() == "keyring":
            # TODO: implement keyring as an auth method
            # keyring.get_password("name_of_app", "password")
            raise NotImplementedError(
                f"{self.auth_method} has not yet been implemented."
            )
        else:
            raise ValueError(
                f"Used authentication method {self.auth_method} is not defined."
            )
        if self.auth_type.lower() == "token":
            assert self.auth_url is not None, (
                "The token authentication method cannot be used "
                "without an authentication URL."
            )
            self.session.auth = TokenAuth(
                username,
                password,
                auth_url=self.auth_url,
                refresh_url=self.refresh_url,
                session=self.session,
                max_login_attempts=self.max_login_attempts,
                token_refresh_delta=self.token_refresh_delta,
            )
        elif self.auth_type.lower() == "basicauth":
            self.session.auth = HTTPBasicAuth(username, password)
        else:
            raise ValueError(
                f"Used authentication type {self.auth_type} is not defined."
            )
