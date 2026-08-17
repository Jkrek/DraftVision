// Central Auth0 configuration, driven entirely by environment variables.
// If the required vars are not set, auth is disabled and the app runs
// fully anonymous (no Auth0Provider, AuthButton hides itself).

export const AUTH0_DOMAIN = process.env.REACT_APP_AUTH0_DOMAIN || '';
export const AUTH0_CLIENT_ID = process.env.REACT_APP_AUTH0_CLIENT_ID || '';
export const AUTH0_AUDIENCE = process.env.REACT_APP_AUTH0_AUDIENCE || '';

export const isAuth0Enabled = Boolean(AUTH0_DOMAIN && AUTH0_CLIENT_ID);
