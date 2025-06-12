import { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import PrediccionForm from './pages/PrediccionForm';
import Historial from './pages/Historial';
import Home from './pages/Home';
import SidebarLayout from './pages/SidebarLayout';

function App() {
  const [userSession, setUserSession] = useState(null);

  if (!userSession) return <LoginPage onLogin={setUserSession} />;

  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={<SidebarLayout usuario={userSession} onLogout={() => setUserSession(null)} />}
        >
          <Route index element={<Home />} />
          <Route path="/prediccion" element={<PrediccionForm usuario={userSession} />} />
          <Route path="/historial" element={<Historial usuario={userSession} />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
