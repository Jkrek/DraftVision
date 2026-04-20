import React from 'react';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import './App.css';
import Navbar from './components/Navbar';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './components/pages/home';
import Services from './components/pages/services';
import Products from './components/pages/Products';
import SignUp from './components/pages/SignUp';
import Auth0ProviderWithHistory from './auth/auth0-provider-with-history';

import PredictionComponent from './components/PredictionComponent';
import Leaderboard from './components/pages/Leaderboard';

function App() {
  return (
    <Router>
      <Auth0ProviderWithHistory>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/services" element={<Services />} />
          <Route path="/products" element={<Products />} />
          <Route path="/sign-up" element={<SignUp />} />
          <Route path="/predict" element={<PredictionComponent />} />
          <Route path="/leaderboard" element={<Leaderboard />} />
        </Routes>
      </Auth0ProviderWithHistory>
    </Router>
  );
}

export default App;
