import datetime
import getpass
import logging
import os
from types import TracebackType
from typing import Optional, Type

import requests
from requests.auth import HTTPBasicAuth


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
        auth_url: str,
        auth_type: str = "token",
        refresh_url: str = None,
        username: str = None,
        auth_method: str = "password",
        token: str = None,
        token_refresh_time_minutes: int = 15,
    ) -> None:
        self.auth_type = auth_type
        self.auth_method = auth_method
        self.auth_url = auth_url
        self.refresh_url = refresh_url
        self.username = username
        self.token_refresh_minutes = token_refresh_time_minutes
        self._user_env_name = "FHIR_USER"
        self._pass_env_name = "FHIR_PASSWORD"
        self.token = token
        self.session = requests.Session()
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
        Authenticate the user and get a token for the current session.

        :return: The token
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
                f"Used authentication method {self.auth_method} is not defined"
            )

        if self.auth_type == "token":
            response = requests.get(f"{self.auth_url}", auth=(username, password))
            response.raise_for_status()
            self.token = response.text
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        elif self.auth_type == "BasicAuth":
            self.session.auth = HTTPBasicAuth(username, password)

    def refresh_token(self, token: str = None) -> None:
        """
        Refresh the token of the current session.

        :param token: If a refresh URL has not been provided, a new token can be provided here as
        parameter
        :return: None
        """
        if token is not None:
            self.token = token
            self.auth_time = datetime.datetime.now()
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        else:
            if self.refresh_url is not None:
                self.auth_time = datetime.datetime.now()
                response = self.session.get(f"{self.refresh_url}")
                response.raise_for_status()
                self.token = response.text
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            else:
                logging.warning(
                    "The token cannot be refreshed because a valid refresh url has not "
                    "been provided."
                )
