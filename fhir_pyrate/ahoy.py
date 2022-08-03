import getpass
import logging
import os
from datetime import timedelta
from types import TracebackType
from typing import Optional, Type, Union

import requests
from requests.auth import HTTPBasicAuth

from fhir_pyrate.util.token_auth import TokenAuth

logger = logging.getLogger(__name__)


class Ahoy:
    """
    Simple authentication class that supports token authentication and BasicAuth.

    :param auth_url: The URL to use for authentication
    :param auth_type: The kind of authentication, for now only "token" and "BasicAuth" are
    supported.
    :param refresh_url:  The URL to use to refresh the token
    :param username: The username to use for the authentication (for the password authentication
    method)
    :param auth_method: The options are [password, env, keyring]:
    password will use the given username as username and ask to input a password;
    env will use the environment variables FHIR_USER and FHIR_PASSWORD;
    keyring will use a keyring [NOT IMPLEMENTED YET]
    :param token: The token that can be used for authentication, if this variable is used then
    the other variables do not need to be specified
    :param max_login_attempts: The maximum number of logins that can be performed
    :param token_refresh_delta: Either a timedelta object that tells us how often the token
    should be refreshed, or a number of minutes; this does not need to be specified for JWT tokens
    that contain the expiry date
    """

    def __init__(
        self,
        auth_url: str = None,
        auth_type: Optional[str] = "token",
        refresh_url: str = None,
        username: str = None,
        auth_method: Optional[str] = "password",
        token: str = None,
        max_login_attempts: int = 5,
        token_refresh_delta: Union[int, timedelta] = None,
    ) -> None:
        self.auth_type = auth_type
        self.auth_method = auth_method
        self.auth_url = auth_url
        self.refresh_url = refresh_url
        self.username = username
        self._user_env_name = "FHIR_USER"
        self._pass_env_name = "FHIR_PASSWORD"
        self.token = token
        self.session = requests.Session()
        self.max_login_attempts = max_login_attempts
        self.token_refresh_delta = token_refresh_delta
        if self.auth_type is not None and self.auth_method is not None:
            self._authenticate()

    def __enter__(self) -> "Ahoy":
        return self

    def close(self) -> None:
        self.session.close()

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None:
        self.close()

    def change_environment_variable_name(
        self, user_env: str = None, pass_env: str = None
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
        assert self.auth_type is not None
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
