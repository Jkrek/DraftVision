import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Auth0Provider } from '@auth0/auth0-react';
import {
  AUTH0_DOMAIN,
  AUTH0_CLIENT_ID,
  AUTH0_AUDIENCE,
  isAuth0Enabled,
} from './config';

const Auth0ProviderWithHistory = ({ children }) => {
  const navigate = useNavigate();

  const onRedirectCallback = (appState) => {
    navigate(appState?.returnTo || window.location.pathname);
  };

  // No Auth0 env vars configured: run the app without auth entirely.
  if (!isAuth0Enabled) {
    return <>{children}</>;
  }

  const authorizationParams = {
    redirect_uri: window.location.origin,
  };
  if (AUTH0_AUDIENCE) {
    authorizationParams.audience = AUTH0_AUDIENCE;
  }

  return (
    <Auth0Provider
      domain={AUTH0_DOMAIN}
      clientId={AUTH0_CLIENT_ID}
      authorizationParams={authorizationParams}
      onRedirectCallback={onRedirectCallback}
      cacheLocation="localstorage"
    >
      {children}
    </Auth0Provider>
  );
};

export default Auth0ProviderWithHistory;
