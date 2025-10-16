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
  const codigo = usuario?.codigo_estudiante || usuario?.codigo;

  useEffect(() => {
    if (!codigo) return;

    const fetchHistorial = async () => {
      const { data, error } = await supabase
        .from('predicciones_estudiantes')
        .select(`
          id,
          anio_ciclo_est,
          fecha_prediccion,
          promedio_predicho,
          nota_real,
          prob_riesgo_no_continuar,
          riesgo_no_continuar
        `)
        .eq('codigo_estudiante', codigo)
        .order('anio_ciclo_est', { ascending: true });

      if (error) {
        console.error('❌ Error al traer historial:', error);
      } else {
        setHistorial(data || []);
        const notas = {};
        (data || []).forEach(item => {
          notas[item.id] = item.nota_real !== null && item.nota_real !== undefined
            ? String(item.nota_real)
            : '';
        });
        setNotasEditables(notas);
      }
    };

    fetchHistorial();
  }, [codigo]);

  const getColorClase = (nota) => {
    const n = Number(nota);
    if (!Number.isFinite(n)) return '';
    if (n >= 16) return styles.verde;
    if (n >= 11) return styles.ambar;
    return styles.rojo;
  };

  const badgeRiesgo = (riesgo) => {
    if (riesgo === 1) return <span className={styles.badgeAlto}>Alto</span>;
    if (riesgo === 0) return <span className={styles.badgeBajo}>Bajo</span>;
    return <span className={styles.badgeNA}>N/A</span>;
  };

  const handleInputChange = (id, valor) => {
    setNotasEditables(prev => ({ ...prev, [id]: valor }));
  };

  const guardarNota = async (id) => {
    const nota = parseFloat(notasEditables[id]);
    if (isNaN(nota) || nota < 0 || nota > 20) {
      alert('Ingresa una nota real válida (0–20).');
      return;
    }

    const { error } = await supabase
      .from('predicciones_estudiantes')
      .update({ nota_real: nota })
      .eq('id', id);

    if (!error) {
      setHistorial(prev =>
        prev.map(item => (item.id === id ? { ...item, nota_real: nota } : item))
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
      [id]: item?.nota_real !== null && item?.nota_real !== undefined ? String(item.nota_real) : ''
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
              <th>Prob. no continuar</th>
              <th>Riesgo</th>
              <th>Nota real</th>
            </tr>
          </thead>
          <tbody>
            {historial.map((item) => {
              const notaProj = Number(item.promedio_predicho);
              const notaProjTxt = Number.isFinite(notaProj) ? notaProj.toFixed(2) : '-';
              const valor = notasEditables[item.id] ?? '';
              const estaEditando = editandoId === item.id;

              const prob = item.prob_riesgo_no_continuar;
              const probTxt = prob !== null && prob !== undefined
                ? `${(prob * 100).toFixed(1)}%`
                : '-';

              const proyectado = (item.anio_ciclo_est ?? 0) + 1;

              return (
                <tr key={item.id}>
                  <td>{item.anio_ciclo_est}</td>
                  <td>{proyectado}</td>
                  <td>{new Date(item.fecha_prediccion).toLocaleDateString()}</td>
                  <td>
                    <span className={`${styles.nota} ${getColorClase(notaProjTxt)}`}>{notaProjTxt}</span>
                  </td>
                  <td>{probTxt}</td>
                  <td>{badgeRiesgo(item.riesgo_no_continuar)}</td>
                  <td>
                    <div className={styles.inputWrapper}>
                      <input
                        type="number"
                        className={styles.inputNota}
                        value={valor}
                        min="0"
                        max="20"
                        step="0.1"
                        onChange={(e) => handleInputChange(item.id, e.target.value)}
                        onFocus={() => setEditandoId(item.id)}
                        onWheel={(e) => e.target.blur()}
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
