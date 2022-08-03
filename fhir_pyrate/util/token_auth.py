import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Union

import jwt
import requests

logger = logging.getLogger(__name__)


class TokenAuth(requests.auth.AuthBase):
    """
    Performs token authentication and handles token refreshes. This class first performs simple
    BasicAuth authentication to obtain a token, and then appends this token to all its requests.
    The token can also be refreshed using a refresh URL.
    The session has a hook that makes sure that the token stays up to date. If the token is a JWT
    token, it is decoded, and we check if the validity of the token is about to be revoked
    (if 75% of the time has already passed). If the token is not a JWT token, it is possible to
    specify a token_refresh_delta (as a timedelta object or as an integer amount of minutes), and
    the token will be refreshed using the refresh URL within that interval.

    :param username: The username of the user
    :param password: The password of the user
    :param auth_url: The URL where the user can be authenticated
    :param refresh_url: A possible refresh URL to get a new token
    :param session: The requests.Session that should be authenticated
    :param max_login_attempts: The maximum number of logins that can be performed
    :param token_refresh_delta: Either a timedelta object that tells us how often the token
    should be refreshed, or a number of minutes; this does not need to be specified for JWT tokens
    that contain the expiry date
    """

    def __init__(
        self,
        username: str,
        password: str,
        auth_url: str,
        refresh_url: str = None,
        session: requests.Session = None,
        max_login_attempts: int = 5,
        token_refresh_delta: Union[int, timedelta] = None,
    ) -> None:
        self._username = username
        self._password = password
        # Future authenticated session
        if session is None:
            self._session = requests.Session()
        else:
            self._session = session
        # Session for handling the tokens
        self._token_session = requests.Session()
        # Add hook to re-authenticate if the token is not valid
        self._session.hooks["response"].append(self._refresh_hook)
        self.auth_url = auth_url
        self.refresh_url = refresh_url
        self._max_login_attempts = max_login_attempts
        self._token_refresh_delta = (
            token_refresh_delta
            if isinstance(token_refresh_delta, timedelta)
            else timedelta(minutes=token_refresh_delta)
            if token_refresh_delta is not None
            else None
        )
        self.token: Optional[str] = None
        self._authenticate()
        self.auth_time = datetime.now()

    def _authenticate(self) -> None:
        """
        Authenticates the user using the authentication URL and sets the token.
        """
        # Authentication to get the token
        response = self._token_session.get(
            f"{self.auth_url}", auth=(self._username, self._password)
        )
        response.raise_for_status()
        self.token = response.text

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        """
        Sets the necessary authentication header of the current request.

        :param r: The prepared request that should be sent
        :return: The prepared request
        """
        r.headers.update({"Authorization": f"Bearer {self.token}"})
        return r

    def is_refresh_required(self) -> bool:
        """
        Computes whether the token should be refreshed according to the given token and to the
        _token_refresh_delta variable.

        :return: Whether the token is about to expire and should thus be refreshed
        """
        # If the token is currently None, then it should always be refreshed
        if self.token is None:
            return True
        try:
            decoded = jwt.decode(
                jwt=self.token,
                options={"verify_signature": False},
            )
            # Get 25 percent of the time we have in total
            refresh_interval = (
                (decoded["exp"] - decoded["iat"]) / 4
                if "exp" in decoded and "iat" in decoded
                else None
            )
            # If there is no expiration time return False
            # If we are already in the last 25% of the time return True
            return refresh_interval is not None and datetime.now().timestamp() > (
                decoded.get("exp") - refresh_interval
            )
        except jwt.exceptions.PyJWTError:
            # If we are here it means that it is not a JWT token
            # If no user limit has been specified, then we do not refresh
            # If it has been specified and the time is almost run out
            return (
                self._token_refresh_delta is not None
                and (datetime.now() - self.auth_time) > self._token_refresh_delta
            )

    def refresh_token(self, token: str = None) -> None:
        """
        Refresh the current session either by logging in again or by refreshing the token.

        :param token: If a refresh URL has not been provided, a new token can be provided here as
        parameter
        """
        logger.info("Refreshing session...")
        if token is not None:
            self.token = token
        elif self.refresh_url is not None:
            response = self._token_session.get(f"{self.refresh_url}")
            # Was not refreshed on time
            if response.status_code == requests.codes.unauthorized:
                self._authenticate()
            else:
                response.raise_for_status()
                self.token = response.text
            self.auth_time = datetime.now()
        else:
            self._authenticate()
            self.auth_time = datetime.now()

    def _refresh_hook(
        self, response: requests.Response, *args: Any, **kwargs: Any
    ) -> Optional[requests.Response]:
        """
        Hook that is called after every request, it checks whether the login was successful and
        if it was not, it either refreshes the token or authenticates the user again.

        :param response: The received response
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        :return: The response of the request that will be sent
        """
        if (
            # If we get an unauthorized or if we should refresh
            response.status_code == requests.codes.unauthorized
            or self.is_refresh_required()
        ):
            # If the state is unauthorized,
            # then we should set how many times we have tried logging in
            if response.status_code == requests.codes.unauthorized:
                if hasattr(response.request, "login_reattempted_times"):
                    response.request.login_reattempted_times += 1  # type: ignore
                    if (
                        response.request.login_reattempted_times  # type: ignore
                        >= self._max_login_attempts
                    ):
                        response.raise_for_status()
                else:
                    response.request.login_reattempted_times = 1  # type: ignore
            # If the token is None, then we were never actually authenticated
            if self.token is None:
                response.raise_for_status()
            else:
                self.token = None
                self.refresh_token()
            # Authenticate and send again
            return self._session.send(self(response.request), **kwargs)
        else:
            # Raise an error for all other cases (if any)
            response.raise_for_status()
        return None
