import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import { isAuth0Enabled } from '../auth/config';

// Renders bare elements on purpose — Navbar.css styles them.
const AuthButtonInner = () => {
  const { loginWithRedirect, logout, isAuthenticated, isLoading, user } = useAuth0();

  if (isLoading) return null;

  if (!isAuthenticated) {
    return <button onClick={() => loginWithRedirect()}>Sign in</button>;
  }

  const label = user?.name || user?.email || 'Account';
  return (
    <>
      <span
        title={user?.email || undefined}
        style={{ fontSize: 13, color: 'var(--color-neutral-300)', whiteSpace: 'nowrap' }}
      >
        {label}
      </span>
      <button
        onClick={() =>
          logout({ logoutParams: { returnTo: window.location.origin } })
        }
      >
        Sign out
      </button>
    </>
  );
};

const AuthButton = () => {
  // Auth0 not configured: hide entirely, app works logged-out.
  if (!isAuth0Enabled) return null;
  return <AuthButtonInner />;
};

export default AuthButton;
