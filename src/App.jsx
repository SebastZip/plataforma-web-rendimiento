import { useEffect, useState } from 'react';
import { supabase } from './supabaseClient';

function App() {
  const [estudiante, setEstudiante] = useState(null);
  const [cargando, setCargando] = useState(true);

  useEffect(() => {
    const obtenerEstudiante = async () => {
      const { data, error } = await supabase
        .from('estudiantes')
        .select('codigo, nombre, carrera')
        .limit(1)
    
  
      console.log('DATA:', data);
      console.log('ERROR:', error);
  
      if (error) {
        console.error('Error al obtener estudiante:', error.message);
      } else {
        setEstudiante(data[0]);
      }
  
      setCargando(false);
    };
  
    obtenerEstudiante();
  }, []);
  
  console.log("¿Estudiante está definido?", estudiante);
  console.log("Código:", estudiante?.codigo);
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#f1f6ff] px-6">
      <div className="bg-white shadow-md rounded-xl p-6 max-w-md w-full">
        <h1 className="text-xl font-bold text-blue-600 mb-4">
          Información del Estudiante
        </h1>
  
        {cargando ? (
  <p className="text-gray-500">Cargando...</p>
) : estudiante ? (
  <div className="space-y-2">
    <p><strong>Código:</strong> {estudiante.codigo}</p>
    <p><strong>Nombre:</strong> {estudiante.nombre}</p>
    <p><strong>Carrera:</strong> {estudiante.carrera}</p>
  </div>
) : (
  <p>No se encontró ningún estudiante.</p>
)}
      </div>
    </div>
  );
}

export default App;

