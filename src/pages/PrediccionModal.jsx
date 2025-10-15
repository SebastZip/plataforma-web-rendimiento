import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import styles from "./PrediccionModal.module.css";

const PrediccionModal = ({ mostrar, onClose, cgpa, ciclo, formData, continuidad }) => {
  const [porcentaje, setPorcentaje] = useState(0);
  const [mostrarResultado, setMostrarResultado] = useState(false);
  const [activarAnimacion, setActivarAnimacion] = useState(false);

  useEffect(() => {
    if (!mostrar || cgpa === null) return;

    setPorcentaje(0);
    setMostrarResultado(false);
    setActivarAnimacion(false);

    const intervalo = setInterval(() => {
      setPorcentaje((prev) => {
        if (prev >= 100) {
          clearInterval(intervalo);
          setTimeout(() => {
            setMostrarResultado(true);
            setTimeout(() => setActivarAnimacion(true), 600);
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
    onClose();
  };

  const getNum = (v) => (v === undefined || v === null || v === "" ? NaN : Number(v));

  // Compara contra el promedio del último ciclo declarado por el alumno
  const compararPonderados = () => {
    const prev = getNum(formData?.promedio_ultima_matricula);
    const pred = getNum(cgpa);
    if (!Number.isFinite(prev) || !Number.isFinite(pred)) return "";
    if (pred > prev) return "📈 El ponderado proyectado supera al último ciclo.";
    if (pred < prev) return "📉 El ponderado proyectado queda por debajo del último ciclo.";
    return "⚖️ El ponderado proyectado se mantiene respecto al último ciclo.";
  };

  // Muestra Regular / Observado según la salida del clasificador
  const renderCondicion = () => {
    if (!continuidad) return null;
    const prob = continuidad?.prob ?? null;      // 0..1
    const observado = continuidad?.riesgo === 1; // 1 = Observado, 0 = Regular

    return (
      <div className={styles.continuidadCard}>
        <h3>🎓 Condición académica proyectada</h3>
        <p className={styles.condicionLinea}>
          Condición:{" "}
          {observado ? (
            <span className={styles.badgeAlto}>Observado</span>
          ) : (
            <span className={styles.badgeBajo}>Regular</span>
          )}
        </p>
        {prob !== null && (
          <p>
            Probabilidad asociada: <b>{(prob * 100).toFixed(1)}%</b>
          </p>
        )}
        <small>
          Nota: “Observado” indica riesgo académico; “Regular” indica continuidad normal.
        </small>
      </div>
    );
  };

  return (
    <AnimatePresence>
      {mostrar && (
        <motion.div
          className={styles.modalOverlay}
          initial={{ y: "-100%", opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: "-100%", opacity: 0 }}
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
                      pathColor: "#2563eb",
                      textColor: "#1e3a8a",
                      trailColor: "#cbd5e1",
                      textSize: "16px",
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
                    initial={{ x: "50%" }}
                    animate={activarAnimacion ? { x: "0%" } : { x: "50%" }}
                    transition={{ duration: 0.8, ease: "easeInOut" }}
                  >
                    <h2 className={styles.tituloFinal}>
                      🎯 Ponderado predicho para el ciclo {ciclo}:
                    </h2>
                    <p className={styles.valorCgpa}>
                      {cgpa !== null ? Number(cgpa).toFixed(2) : "--"}
                    </p>
                    <p className={styles.comparacion}>{compararPonderados()}</p>
                  </motion.div>
                </motion.div>

                {renderCondicion()}

                <div className={styles.footerAcciones}>
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
