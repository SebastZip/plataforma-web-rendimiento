import { useEffect, useState } from 'react';
import { createClient } from '@supabase/supabase-js';
import styles from './Historial.module.css';

const supabase = createClient(
  "https://ghpihkczhzoaydrceflm.supabase.co",
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdocGloa2N6aHpvYXlkcmNlZmxtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUwNTMzNjIsImV4cCI6MjA2MDYyOTM2Mn0.RUo6qktKAQcmgBYWLa0Lq_pE1UdLB2KS1nWLxr-HaIM"
);

const Historial = ({ usuario }) => {
  const [historial, setHistorial] = useState([]);
  const [editandoId, setEditandoId] = useState(null);
  const [notasEditables, setNotasEditables] = useState({});

  useEffect(() => {
    if (!usuario || !usuario.codigo_estudiante) return;

    const fetchHistorial = async () => {
      const { data, error } = await supabase
        .from('predicciones_estudiantes')
        .select('id, current_semester, semestre_a_predecir, fecha_prediccion, ponderado_predecido, nota_real')
        .eq('codigo_estudiante', usuario.codigo_estudiante)
        .order('current_semester', { ascending: true });

      if (error) {
        console.error('❌ Error al traer historial:', error);
      } else {
        setHistorial(data);
        const notas = {};
        data.forEach(item => {
          notas[item.id] = item.nota_real !== null ? item.nota_real.toString() : '';
        });
        setNotasEditables(notas);
      }
    };

    fetchHistorial();
  }, [usuario]);

  const getColorClase = (nota) => {
    if (nota >= 16) return styles.verde;
    if (nota >= 11) return styles.ambar;
    return styles.rojo;
  };

  const handleInputChange = (id, valor) => {
    setNotasEditables(prev => ({
      ...prev,
      [id]: valor
    }));
  };

  const guardarNota = async (id) => {
    const nota = parseFloat(notasEditables[id]);
    if (isNaN(nota)) return;

    const { error } = await supabase
      .from('predicciones_estudiantes')
      .update({ nota_real: nota })
      .eq('id', id);

    if (!error) {
      setHistorial(prev =>
        prev.map(item => item.id === id ? { ...item, nota_real: nota } : item)
      );
      setEditandoId(null);
    } else {
      console.error('❌ Error al guardar nota:', error);
    }
  };

  const cancelarEdicion = (id) => {
    const item = historial.find(row => row.id === id);
    setNotasEditables(prev => ({
      ...prev,
      [id]: item.nota_real !== null ? item.nota_real.toString() : ''
    }));
    setEditandoId(null);
  };

  return (
    <div className={styles.container}>
      <h2 className={styles.titulo}>📊 Historial de Predicciones</h2>

      {historial.length === 0 ? (
        <p className={styles.mensajeVacio}>Aún no tienes predicciones registradas.</p>
      ) : (
        <table className={`${styles.tabla} ${styles.fadeIn}`}>
          <thead>
            <tr>
              <th>Semestre actual</th>
              <th>Semestre proyectado</th>
              <th>Fecha</th>
              <th>Nota proyectada</th>
              <th>Nota real</th>
            </tr>
          </thead>
          <tbody>
            {historial.map((item) => {
              const nota = Number(item.ponderado_predecido).toFixed(2);
              const valor = notasEditables[item.id] ?? '';
              const estaEditando = editandoId === item.id;

              return (
                <tr key={item.id}>
                  <td>{item.current_semester}</td>
                  <td>{item.semestre_a_predecir}</td>
                  <td>{new Date(item.fecha_prediccion).toLocaleDateString()}</td>
                  <td>
                    <span className={`${styles.nota} ${getColorClase(nota)}`}>{nota}</span>
                  </td>
                  <td>
                    <div className={styles.inputWrapper}>
                      <input
                        type="number"
                        className={styles.inputNota}
                        value={valor}
                        onChange={(e) => handleInputChange(item.id, e.target.value)}
                        onFocus={() => setEditandoId(item.id)}
                      />
                      {estaEditando && (
                        <div className={styles.botonesAccion}>
                          <button onClick={() => guardarNota(item.id)} className={styles.botonCheck}>✅</button>
                          <button onClick={() => cancelarEdicion(item.id)} className={styles.botonCancel}>❌</button>
                        </div>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default Historial;
