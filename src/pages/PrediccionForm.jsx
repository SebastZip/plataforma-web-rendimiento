import { useState } from "react";
import styles from "./PrediccionForm.module.css";
import { supabase } from "../supabaseClient";
import PrediccionModal from "./PrediccionModal";

const API_BASE =
  import.meta.env.VITE_API_BASE || "https://plataforma-web-rendimiento-i2x4.onrender.com";

/** ===== Campos del formulario =====
 * Requeridos (alimentan directamente al modelo actual)
 */
const camposRequeridos = [
  ["promedio_ultima_matricula", "📊 Promedio ponderado ÚLTIMO ciclo (0–20)", { min: 0, max: 20, step: "0.01", required: true }],
  ["semestre_actual", "📘 Semestre actual (1–10)", { min: 1, max: 10, step: "1", required: true }],
  ["num_periodo_acad_matric", "🧾 Nº de matrículas cursadas", { min: 0, max: 30, step: "1", required: true }],
  ["ultimo_periodo_matriculado", "🗓️ Último periodo matriculado (AAAAT)", { min: 20101, max: 20999, step: "1", required: true, inputMode: "numeric", pattern: "[0-9]*" }],
  ["anio_ingreso", "🎓 Año de ingreso a la FISI", { min: 2005, max: 2035, step: "1", required: true }],
];

/** Opcionales (para investigación / futuro reentrenamiento) */
const camposOpcionales = [
  ["asistencia_promedio_pct", "📊 Asistencia promedio (%) — opcional", { min: 0, max: 100, step: "0.1" }],
  ["horas_estudio_diarias", "📘 Horas de estudio diarias — opcional", { min: 0, max: 24, step: "0.1" }],
  ["horas_redes_diarias", "📱 Horas en redes diarias — opcional", { min: 0, max: 24, step: "0.1" }],
  ["horas_habilidades_diarias", "💻 Horas en habilidades/actividades — opcional", { min: 0, max: 24, step: "0.1" }],
  ["ingreso_familiar_mensual_soles", "💰 Ingreso familiar mensual (S/.) — opcional", { min: 0, step: "1" }],
];

/** Binarios (opcionales) */
const camposSiNo = [
  ["estado_observado", "⚠️ ¿Alumno observado? (desaprobaste un curso más de dos veces)", ["Sí", "No"]],
  ["desaprobo_alguna_asignatura", "❌ ¿Desaprobaste alguna asignatura el ciclo anterior?", ["Sí", "No"]],
  ["beca_subvencion_economica", "🏅 ¿Cuentas con beca o subvención económica?", ["Sí", "No"]],
  ["planea_matricularse_prox_ciclo", "🧭 ¿Planeas matricularte el próximo ciclo académico?", ["Sí", "No"]],
];

/** Utilidades */
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const validarPeriodo = (v) => {
  const n = Number(v);
  const year = Math.floor(n / 10);
  const term = n % 10;
  return year >= 2010 && year <= 2099 && [0, 1, 2].includes(term);
};

const PrediccionForm = ({ usuario }) => {
  const [formData, setFormData] = useState({ codigo_estudiante: usuario.codigo });
  const [resultadoReg, setResultadoReg] = useState(null);
  const [resultadoCls, setResultadoCls] = useState(null);
  const [cicloObjetivo, setCicloObjetivo] = useState(null);
  const [mostrarModal, setMostrarModal] = useState(false);
  const [cargando, setCargando] = useState(false);

  const onChange = (e) => {
    const { name, value } = e.target;
    const num = Number(value);
    setFormData((s) => ({
      ...s,
      [name]: Number.isFinite(num) ? num : value,
    }));
  };

  const onChangeSiNo = (e) => {
    const { name, value } = e.target;
    setFormData((s) => ({ ...s, [name]: value })); // "Sí"/"No" → se mapea antes de insertar
  };

  /** "Sí/No" → boolean */
  const mapSiNoABool = (v) => (v === "Sí" ? true : v === "No" ? false : v);

  /** Construye la fila para Supabase */
  const construirFilaSupabase = () => {
    const fila = { ...formData, codigo_estudiante: String(usuario.codigo).trim() };

    // mapear binarios
    camposSiNo.forEach(([name]) => {
      if (name in fila && fila[name] !== "") fila[name] = mapSiNoABool(fila[name]);
    });

    // garantizar numéricos
    [...camposRequeridos, ...camposOpcionales].forEach(([name]) => {
      if (name in fila && fila[name] !== "") fila[name] = Number(fila[name]);
    });

    // tipos clave
    if (fila.semestre_actual !== undefined) fila.semestre_actual = Number(fila.semestre_actual);
    if (fila.ultimo_periodo_matriculado !== undefined) fila.ultimo_periodo_matriculado = Number(fila.ultimo_periodo_matriculado);

    return fila;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setCargando(true);
    setResultadoReg(null);
    setResultadoCls(null);
    setMostrarModal(true);

    try {
      const fila = construirFilaSupabase();

      // Validación mínima (solo requeridos)
      if (!fila.semestre_actual) throw new Error("Debes ingresar el semestre actual.");
      if (fila.promedio_ultima_matricula == null) throw new Error("Falta el promedio del último ciclo.");
      if (!validarPeriodo(fila.ultimo_periodo_matriculado)) {
        throw new Error("Último periodo matriculado inválido (usa AAAAT con T ∈ {0,1,2}).");
      }
      if (fila.anio_ingreso < 2005 || fila.anio_ingreso > 2035) {
        throw new Error("Año de ingreso fuera de rango (2005–2035).");
      }

      // ✅ UPSERT para no chocar con la unique constraint
      const { data: upserted, error: errUp } = await supabase
        .from("predicciones_estudiantes")
        .upsert(fila, { onConflict: "codigo_estudiante,semestre_actual" })
        .select();

      if (errUp) throw errUp;

      // Si tu BD setea semestre_proyectado por trigger/función, intenta leerlo del upsert
      const rec = Array.isArray(upserted) ? upserted[0] : null;
      setCicloObjetivo(rec?.semestre_proyectado ?? fila.semestre_actual + 1);

      // Llamar a ambas APIs (guardan en la fila con save=true)
      const [resR, resC] = await Promise.all([
        fetch(`${API_BASE}/predict/regresion/${fila.codigo_estudiante}?save=true`),
        fetch(`${API_BASE}/predict/continuidad/${fila.codigo_estudiante}?save=true`),
      ]);

      const jsonR = await resR.json();
      const jsonC = await resC.json();

      if (!resR.ok) throw new Error(jsonR.detail || "Error regresión");
      if (!resC.ok) throw new Error(jsonC.detail || "Error continuidad");

      setResultadoReg(jsonR.promedio_predicho);
      setResultadoCls({ prob: jsonC.prob_riesgo, riesgo: jsonC.riesgo });
    } catch (err) {
      console.error(err);
      alert("❌ Error durante el registro o la predicción: " + (err?.message || ""));
    } finally {
      setCargando(false);
    }
  };

  return (
    <div className={styles.wrapper}>
      <h2 className={styles.subtitulo}>🧠 Cuestionario de Predicción</h2>

      <form className={styles.formulario} onSubmit={handleSubmit}>
        <div className={styles.gridInputs}>
          {/* Requeridos */}
          {camposRequeridos.map(([name, label, extra]) => (
            <div key={name} className={styles.inputCard}>
              <label htmlFor={name}>{label}</label>
              <input
                id={name}
                type="number"
                name={name}
                value={formData[name] ?? ""}
                onChange={onChange}
                {...extra}
                onWheel={(e) => e.currentTarget.blur()}
              />
            </div>
          ))}

          {/* Opcionales */}
          {camposOpcionales.map(([name, label, extra]) => (
            <div key={name} className={styles.inputCard}>
              <label htmlFor={name}>{label}</label>
              <input
                id={name}
                type="number"
                name={name}
                value={formData[name] ?? ""}
                onChange={onChange}
                {...extra}
                onWheel={(e) => e.currentTarget.blur()}
              />
            </div>
          ))}

          {/* Sí/No (opcionales) */}
          {camposSiNo.map(([name, label, options]) => (
            <div key={name} className={styles.inputCard}>
              <label htmlFor={name}>{label}</label>
              <select
                id={name}
                name={name}
                value={formData[name] ?? ""}
                onChange={onChangeSiNo}
              >
                <option hidden value="">
                  Selecciona una opción
                </option>
                {options.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
            </div>
          ))}
        </div>

        <button className={styles.botonVerde} type="submit" disabled={cargando}>
          {cargando ? "Procesando..." : "✅ Finalizar cuestionario"}
        </button>
      </form>

      <PrediccionModal
        mostrar={mostrarModal}
        onClose={() => setMostrarModal(false)}
        cgpa={resultadoReg}
        ciclo={cicloObjetivo}
        formData={formData}
        continuidad={resultadoCls}
      />
    </div>
  );
};

export default PrediccionForm;
