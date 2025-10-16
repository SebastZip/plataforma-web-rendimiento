import { useState } from "react";
import styles from "./PrediccionForm.module.css";
import { supabase } from "../supabaseClient";
import PrediccionModal from "./PrediccionModal";

const API_BASE =
  import.meta.env.VITE_API_BASE || "https://plataforma-web-rendimiento-i2x4.onrender.com";

/** ===== Catálogos restringidos ===== */
const programas = [
  "E.P. de Ingenieria de Sistemas",
  "E.P. de Ingenieria de Software",
];

const sexos = [
  { label: "Femenino", value: "F" },
  { label: "Masculino", value: "M" },
];

/** ===== Campos del formulario (OBLIGATORIOS, nombres exactos) ===== */
const camposObligatorios = [
  ["promedio_ultima_matricula", "📊 Promedio ponderado ÚLTIMO ciclo (0–20)", { min: 0, max: 20, step: "0.01", required: true }],
  ["anio_ciclo_est", "📘 Semestre actual (1–10)", { min: 1, max: 10, step: "1", required: true }],
  ["num_periodo_acad_matric", "🧾 Nº de matrículas cursadas", { min: 0, max: 30, step: "1", required: true }],
  // BD: NOT NULL (AAAAT con T∈{0,1,2})
  ["ultimo_periodo_matriculado", "🗓️ Último periodo matriculado (AAAAT)", { min: 20101, max: 20999, step: "1", required: true, inputMode: "numeric", pattern: "[0-9]*" }],
  ["edad_en_ingreso", "🎯 Edad al ingresar a la FISI", { min: 15, max: 70, step: "1", required: true }],
  ["anio_ingreso", "🎓 Año de ingreso a la FISI", { min: 2005, max: 2035, step: "1", required: true }],
];

/** ===== Opcionales ===== */
const camposOpcionales = [
  ["asistencia_promedio_pct", "📊 Asistencia promedio (%) — opcional", { min: 0, max: 100, step: "0.1" }],
  ["horas_estudio_diarias", "📘 Horas de estudio diarias — opcional", { min: 0, max: 24, step: "0.1" }],
  ["horas_habilidades_diarias", "💻 Horas en habilidades/actividades — opcional", { min: 0, max: 24, step: "0.1" }],
  ["ingreso_familiar_mensual_soles", "💰 Ingreso familiar mensual (S/.) — opcional", { min: 0, step: "1" }],
];

/** Utilidades */
const validarRango = (v, min, max) => Number.isFinite(v) && v >= min && v <= max;
const validarPeriodoAAAAT = (v) => {
  const n = Number(v);
  const year = Math.floor(n / 10);
  const term = n % 10;
  return year >= 2010 && year <= 2099 && [0, 1, 2].includes(term);
};

const PrediccionForm = ({ usuario }) => {
  const [formData, setFormData] = useState({
    codigo_estudiante: usuario.codigo,
    programa: "", // requerido por el modelo (2 opciones)
    sexo: "",     // requerido por el modelo ("M" o "F")
  });
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

  const onSelectPrograma = (e) => {
    setFormData((s) => ({ ...s, programa: e.target.value }));
  };

  const onSelectSexo = (e) => {
    setFormData((s) => ({ ...s, sexo: e.target.value })); // "M" o "F"
  };

  /** Construye la fila para Supabase (y predicción) */
  const construirFilaSupabase = () => {
    const fila = {
      ...formData,
      codigo_estudiante: String(usuario.codigo).trim(),
    };

    // Garantizar numéricos en obligatorios/opcionales
    [...camposObligatorios, ...camposOpcionales].forEach(([name]) => {
      if (fila[name] !== undefined && fila[name] !== "") {
        const num = Number(fila[name]);
        fila[name] = Number.isFinite(num) ? num : fila[name];
      }
    });

    // Validar programa dentro del catálogo
    if (!programas.includes(fila.programa)) {
      throw new Error("Programa inválido. Debe ser 'E.P. de Ingenieria de Sistemas' o 'E.P. de Ingenieria de Software'.");
    }

    // Validar sexo M/F
    if (!["M", "F"].includes(fila.sexo)) {
      throw new Error("Sexo inválido. Solo se admite 'M' o 'F'.");
    }

    // Tipos clave
    if (fila.ultimo_periodo_matriculado !== undefined) {
      fila.ultimo_periodo_matriculado = Number(fila.ultimo_periodo_matriculado);
    }

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

      // Validación mínima de obligatorios
      if (!validarRango(fila.promedio_ultima_matricula, 0, 20)) {
        throw new Error("Promedio del último ciclo fuera de rango (0–20).");
      }
      if (!validarRango(fila.anio_ciclo_est, 1, 10)) {
        throw new Error("Semestre actual (anio_ciclo_est) fuera de rango (1–10).");
      }
      if (!Number.isInteger(fila.num_periodo_acad_matric) || fila.num_periodo_acad_matric < 0) {
        throw new Error("Número de matrículas inválido.");
      }
      if (!validarPeriodoAAAAT(fila.ultimo_periodo_matriculado)) {
        throw new Error("Último periodo matriculado inválido (usa AAAAT con T ∈ {0,1,2}).");
      }
      if (!validarRango(fila.edad_en_ingreso, 15, 70)) {
        throw new Error("Edad al ingresar inválida (15–70).");
      }
      if (!validarRango(fila.anio_ingreso, 2005, 2035)) {
        throw new Error("Año de ingreso fuera de rango (2005–2035).");
      }

      // ✅ UPSERT (clave única: codigo_estudiante + anio_ciclo_est)
      const { data: upserted, error: errUp } = await supabase
        .from("predicciones_estudiantes")
        .upsert(fila, { onConflict: "codigo_estudiante,anio_ciclo_est" })
        .select();

      if (errUp) throw errUp;

      const rec = Array.isArray(upserted) ? upserted[0] : null;
      setCicloObjetivo((rec?.anio_ciclo_est ?? fila.anio_ciclo_est) + 1);

      // Predicción y guardado (por código; el backend usa los campos guardados)
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
          {/* Obligatorios (nombres exactos) */}
          {camposObligatorios.map(([name, label, extra]) => (
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

          {/* Programa (solo 2 opciones) */}
          <div className={styles.inputCard}>
            <label htmlFor="programa">🏫 Programa académico</label>
            <select
              id="programa"
              name="programa"
              value={formData.programa ?? ""}
              onChange={onSelectPrograma}
              required
            >
              <option hidden value="">Selecciona tu programa</option>
              {programas.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>

          {/* Sexo (solo M/F) */}
          <div className={styles.inputCard}>
            <label htmlFor="sexo">🧑 Sexo</label>
            <select
              id="sexo"
              name="sexo"
              value={formData.sexo ?? ""}
              onChange={onSelectSexo}
              required
            >
              <option hidden value="">Selecciona</option>
              {sexos.map((sx) => (
                <option key={sx.value} value={sx.value}>{sx.label}</option>
              ))}
            </select>
          </div>

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
