import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import styles from './PrediccionModal.module.css';

const PrediccionModal = ({ mostrar, onClose, cgpa, ciclo, formData, continuidad }) => {
  const [porcentaje, setPorcentaje] = useState(0);
  const [mostrarResultado, setMostrarResultado] = useState(false);
  const [activarAnimacion, setActivarAnimacion] = useState(false);
  const [mostrarRecomendaciones, setMostrarRecomendaciones] = useState(false);

  useEffect(() => {
    if (!mostrar || cgpa === null) return;

    setPorcentaje(0);
    setMostrarResultado(false);
    setActivarAnimacion(false);
    setMostrarRecomendaciones(false);

    const intervalo = setInterval(() => {
      setPorcentaje((prev) => {
        if (prev >= 100) {
          clearInterval(intervalo);
          setTimeout(() => {
            setMostrarResultado(true);
            setTimeout(() => {
              setActivarAnimacion(true);
              setTimeout(() => setMostrarRecomendaciones(true), 800);
            }, 1000);
          }, 300);
          return 100;
        }
        return prev + 1;
      });
    }, 20);

    return () => clearInterval(intervalo);
  }, [mostrar, cgpa]);

  const handleClose = () => {
    setPorcentaje(0);
    setMostrarResultado(false);
    setActivarAnimacion(false);
    setMostrarRecomendaciones(false);
    onClose();
  };

  // ---- NUEVOS nombres de campos ----
  const getNum = (v) => (v === undefined || v === null || v === '' ? NaN : Number(v));

  const generarRecomendaciones = () => {
    const recomendaciones = [];
    const asistencia = getNum(formData?.asistencia_promedio_pct);
    const redes = getNum(formData?.horas_redes_diarias);
    const skill = getNum(formData?.horas_habilidades_diarias);
    const study = getNum(formData?.horas_estudio_diarias);

    if (Number.isFinite(asistencia) && asistencia < 70) recomendaciones.push("📚 Mejora tu asistencia (menos del 70%).");
    if (Number.isFinite(redes) && redes > 5) recomendaciones.push("🚫 Reduce el tiempo en redes sociales (más de 5 h).");
    if (Number.isFinite(skill) && skill < 2) recomendaciones.push("🧠 Suma tiempo a habilidades (menos de 2 h).");
    if (Number.isFinite(study) && study < 2) recomendaciones.push("📘 Aumenta horas de estudio (menos de 2 h).");

    return recomendaciones.length ? recomendaciones : ["✅ ¡Sigue así! Mantén tus hábitos actuales."];
  };

  const compararPonderados = () => {
    const prev = getNum(formData?.sgpa_previo);
    const pred = getNum(cgpa);
    if (!Number.isFinite(prev) || !Number.isFinite(pred)) return "";
    if (pred > prev) return "📈 Tu ponderado proyectado supera al ciclo anterior.";
    if (pred < prev) return "📉 Tu ponderado proyectado queda por debajo del ciclo anterior.";
    return "⚖️ Tu ponderado proyectado se mantiene estable vs. el ciclo anterior.";
  };

  const renderContinuidad = () => {
    if (!continuidad) return null;
    const prob = continuidad?.prob ?? null;  // 0..1
    const riesgo = continuidad?.riesgo === 1; // bool
    return (
      <div className={styles.continuidadCard}>
        <h3>🧭 Continuidad académica (proyección)</h3>
        {prob !== null && (
          <p>
            Prob. de <strong>no continuar</strong>: <b>{(prob * 100).toFixed(1)}%</b>
            {" "}
            {riesgo ? <span className={styles.badgeAlto}>Riesgo alto</span>
                    : <span className={styles.badgeBajo}>Riesgo bajo</span>}
          </p>
        )}
        <small>Interpretación: si el riesgo es alto, refuerza hábitos y busca consejería académica.</small>
      </div>
    );
  };

  return (
    <AnimatePresence>
      {mostrar && (
        <motion.div
          className={styles.modalOverlay}
          initial={{ y: '-100%', opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: '-100%', opacity: 0 }}
          transition={{ duration: 0.4 }}
        >
          <div className={styles.modalContenido}>
            {!mostrarResultado ? (
              <div className={styles.loaderContainer}>
                <div className={styles.loaderCircular}>
                  <CircularProgressbar
                    value={porcentaje}
                    text={`${porcentaje}%`}
                    styles={buildStyles({
                      pathColor: '#2563eb',
                      textColor: '#1e3a8a',
                      trailColor: '#cbd5e1',
                      textSize: '16px'
                    })}
                  />
                  <p className={styles.textoCargando}>Generando predicción...</p>
                </div>
              </div>
            ) : (
              <div className={styles.resultadoYRecomendaciones}>
                <motion.div
                  className={styles.resultadoFinal}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: mostrarResultado ? 1 : 0 }}
                  transition={{ duration: 0.8 }}
                >
                  <motion.div
                    initial={{ x: '50%' }}
                    animate={activarAnimacion ? { x: '0%' } : { x: '50%' }}
                    transition={{ duration: 0.8, ease: 'easeInOut' }}
                  >
                    <h2 className={styles.tituloFinal}>
                      🎯 Ponderado predicho para el ciclo {ciclo}:
                    </h2>
                    <p className={styles.valorCgpa}>
                      {cgpa !== null ? Number(cgpa).toFixed(2) : '--'}
                    </p>
                    <p className={styles.comparacion}>{compararPonderados()}</p>
                  </motion.div>
                </motion.div>

                {renderContinuidad()}

                <div
                  className={styles.recomendaciones}
                  style={{ opacity: mostrarRecomendaciones ? 1 : 0 }}
                >
                  <h3>📑 Recomendaciones:</h3>
                  <ul>
                    {generarRecomendaciones().map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                  <button className={styles.botonCerrar} onClick={handleClose}>
                    Cerrar
                  </button>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default PrediccionModal;
