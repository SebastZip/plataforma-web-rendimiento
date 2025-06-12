import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import styles from './PrediccionModal.module.css';

const PrediccionModal = ({ mostrar, onClose, cgpa, ciclo, formData }) => {
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
              setTimeout(() => {
                setMostrarRecomendaciones(true);
              }, 800);
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
    onClose(); // ejecutar cierre real desde el padre
  };

  const generarRecomendaciones = () => {
    const recomendaciones = [];
    const asistencia = parseFloat(formData.average_attendance);
    const redes = parseFloat(formData.social_media_time);
    const skill = parseFloat(formData.skill_time);
    const study = parseFloat(formData.study_time);

    if (asistencia < 70) recomendaciones.push("📚 Mejora tu asistencia a clases (menos del 70%)");
    if (redes > 5) recomendaciones.push("🚫 Reduce el tiempo en redes sociales (más de 5 h)");
    if (skill < 2) recomendaciones.push("🧠 Dedica más tiempo al desarrollo de habilidades (menos de 2 h)");
    if (study < 2) recomendaciones.push("📘 Aumenta tus horas de estudio diario (menos de 2 h)");

    return recomendaciones;
  };

  const compararPonderados = () => {
    const prev = parseFloat(formData.previous_sgpa);
    const pred = parseFloat(cgpa);
    if (pred > prev) return "📈 Tu ponderado ha aumentado respecto al ciclo anterior.";
    if (pred < prev) return "📉 Tu ponderado ha disminuido respecto al ciclo anterior.";
    return "⚖️ Tu ponderado se mantiene estable.";
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
                      🎯 El ponderado predicho para el ciclo {ciclo} es:
                    </h2>
                    <p className={styles.valorCgpa}>
                      {cgpa !== null ? cgpa.toFixed(2) : '--'}
                    </p>
                    <p className={styles.comparacion}>{compararPonderados()}</p>
                  </motion.div>
                </motion.div>

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
