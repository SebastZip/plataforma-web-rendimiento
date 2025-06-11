import { useEffect, useState } from 'react';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_ANON_KEY
);

const Historial = ({ usuario }) => {
  const [historial, setHistorial] = useState([]);

  useEffect(() => {
    const fetchHistorial = async () => {
      const { data, error } = await supabase
        .from('predicciones_estudiantes')
        .select('semestre_actual, semestre_a_predecir, fecha_prediccion, ponderado_predecido')
        .eq('codigo_estudiante', usuario.codigo)
        .order('semestre_actual', { ascending: true });

      if (error) console.error('Error:', error);
      else setHistorial(data);
    };

    fetchHistorial();
  }, [usuario]);

  return (
    <div>
      <h2>📊 Historial de Predicciones</h2>
      <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '1rem' }}>
        <thead>
          <tr>
            <th>Semestre actual</th>
            <th>Semestre proyectado</th>
            <th>Fecha</th>
            <th>Nota proyectada</th>
          </tr>
        </thead>
        <tbody>
          {historial.map((item, index) => (
            <tr key={index}>
              <td>{item.semestre_actual}</td>
              <td>{item.semestre_a_predecir}</td>
              <td>{new Date(item.fecha_prediccion).toLocaleDateString()}</td>
              <td>{Number(item.ponderado_predecido).toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Historial;
