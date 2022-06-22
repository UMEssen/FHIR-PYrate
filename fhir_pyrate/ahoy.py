import datetime
import getpass
import logging
import os
from types import TracebackType
from typing import Any, Optional, Type

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)


class Ahoy:
    """
    Simple authentication class.

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
    :param token_refresh_time_minutes: The number of minutes after which the token should be
    refreshed
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
        self.auth_time = None
        self.login_reattempted_times = 0
        self.max_login_attempts = max_login_attempts
        self.session.hooks["response"].append(self._refresh_hook)
        if self.auth_type is not None and self.auth_method is not None:
            self._authenticate()
            self.auth_time = datetime.datetime.now()

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
        if self.auth_method == "password":
            username = self.username
            password = getpass.getpass()
        elif self.auth_method == "env":
            username = os.environ[self._user_env_name]
            password = os.environ[self._pass_env_name]
        elif self.auth_method == "keyring":
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
            response = requests.get(f"{self.auth_url}", auth=(username, password))
            response.raise_for_status()
            self.token = response.text
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        elif self.auth_type.lower() == "basicauth":
            self.session.auth = HTTPBasicAuth(username, password)
        else:
            raise ValueError(
                f"Used authentication type {self.auth_type} is not defined."
            )

    def refresh_session(self, token: str = None) -> None:
        """
        Refresh the current session either by logging in again or by refreshing the token.

        :param token: If a refresh URL has not been provided, a new token can be provided here as
        parameter
        """
        logger.info("Refreshing session...")
        if token is not None:
            self.token = token
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            self.auth_time = datetime.datetime.now()
        elif self.refresh_url is not None:
            response = self.session.get(f"{self.refresh_url}")
            response.raise_for_status()
            self.token = response.text
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            self.auth_time = datetime.datetime.now()
        elif self.auth_type is not None and self.auth_method is not None:
            self._authenticate()
            self.auth_time = datetime.datetime.now()

    def _refresh_hook(
        self, response: requests.Response, *args: Any, **kwargs: Any
    ) -> Optional[requests.Response]:
        if self.login_reattempted_times >= self.max_login_attempts:
            response.raise_for_status()
        elif response.status_code == requests.codes.unauthorized:
            self.login_reattempted_times += 1
            self.token = None
            self.refresh_session()
            response.request.headers.update(self.session.headers)
            # TODO: Untested with BasicAuth
            response.request.prepare_auth(self.session.auth)
            return self.session.send(response.request, **kwargs)
        elif response.status_code == requests.codes.ok:
            self.login_reattempted_times = 0
        return None
