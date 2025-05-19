import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { FormularioPrediccion } from './pages/PrediccionForm.jsx';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<FormularioPrediccion />} />
      </Routes>
    </Router>
  );
}

export default App;

