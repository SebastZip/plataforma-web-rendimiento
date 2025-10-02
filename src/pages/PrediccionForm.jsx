import { useState } from "react";
import styles from "./PrediccionForm.module.css";
import { supabase } from "../supabaseClient";
import PrediccionModal from "./PrediccionModal";

const API_BASE = import.meta.env.VITE_API_BASE || "https://plataforma-web-rendimiento-i2x4.onrender.com";

/** ---- Campos (coinciden con tu tabla nueva) ---- **/
const camposNumero = [
  ["sgpa_previo", "📊 Promedio ponderado PREVIO (0–20)", { min: 0, max: 20, step: "0.01" }],
  ["cgpa_actual", "📊 Promedio ponderado ACTUAL (0–20)", { min: 0, max: 20, step: "0.01" }],
  ["creditos_completados", "✅ Créditos completados (SUG: del SUM)", { min: 0, max: 300, step: "1" }],
  ["semestre_actual", "📘 Semestre actual (1–10)", { min: 1, max: 10, step: "1" }],
  ["asistencia_promedio_pct", "📊 Asistencia promedio (%)", { min: 0, max: 100, step: "0.1" }],
  ["horas_estudio_diarias", "📘 Horas de estudio diarias", { min: 0, max: 24, step: "0.1" }],
  ["horas_redes_diarias", "📱 Horas diarias en redes sociales", { min: 0, max: 24, step: "0.1" }],
  ["horas_habilidades_diarias", "💻 Horas diarias en habilidades/actividades (cursos, talleres…)", { min: 0, max: 24, step: "0.1" }],
  ["ingreso_familiar_mensual_soles", "💰 Ingreso familiar mensual (S/.)", { min: 0, step: "1" }],
  ["edad", "📅 Edad", { min: 15, max: 80, step: "1" }],
  ["anio_egreso_secundaria", "🎓 Año de egreso de secundaria", { min: 2000, max: 2035, step: "1" }],
];

const camposSiNo = [
  ["estado_observado", "⚠️ ¿Alumno observado? (desaprobaste un curso más de dos veces)", ["Sí", "No"]],
  ["beca_subvencion_economica", "🏅 ¿Cuentas con beca o subvención económica?", ["Sí", "No"]],
  ["planea_matricularse_prox_ciclo", "🧭 ¿Planeas matricularte el próximo ciclo académico?", ["Sí", "No"]],
  // Si luego modelas HE01:
  // ["desaprobo_alguna_asignatura", "❌ ¿Desaprobaste alguna asignatura el ciclo anterior?", ["Sí","No"]],
];

/** Utilidad para pausar */
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const PrediccionForm = ({ usuario }) => {
  const [formData, setFormData] = useState({ codigo_estudiante: usuario.codigo });
  const [resultadoReg, setResultadoReg] = useState(null);
  const [resultadoCls, setResultadoCls] = useState(null);
  const [cicloObjetivo, setCicloObjetivo] = useState(null);
  const [mostrarModal, setMostrarModal] = useState(false);
  const [cargando, setCargando] = useState(false);

  const onChange = (e) => {
    const { name, value } = e.target;
    // para mantener números realmente numéricos
    const num = Number(value);
    setFormData((s) => ({
      ...s,
      [name]: Number.isFinite(num) ? num : value,
    }));
  };

  const onChangeSiNo = (e) => {
    const { name, value } = e.target;
    setFormData((s) => ({ ...s, [name]: value })); // guardamos "Sí"/"No", mapeamos antes de insertar
  };

  /** Convierte "Sí/No" → boolean para Supabase */
  const mapSiNoABool = (v) => (v === "Sí" ? true : v === "No" ? false : v);

  /** Prepara payload para Supabase (coincide con columnas) */
  const construirFilaSupabase = () => {
    const fila = { ...formData, codigo_estudiante: usuario.codigo };

    camposSiNo.forEach(([name]) => {
      if (name in fila) fila[name] = mapSiNoABool(fila[name]);
    });

    // asegúrate de que numéricos queden como números
    camposNumero.forEach(([name]) => {
      if (name in fila) fila[name] = Number(fila[name]);
    });

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

      // Validación mínima
      if (!fila.semestre_actual) throw new Error("Debes ingresar el semestre actual.");
      if (!fila.sgpa_previo && fila.sgpa_previo !== 0) throw new Error("Falta SGPA previo.");

      // 1) ¿Ya existe fila para (codigo, semestre_actual)?
      const { data: existentes, error: errSel } = await supabase
        .from("predicciones_estudiantes")
        .select("id")
        .eq("codigo_estudiante", usuario.codigo)
        .eq("semestre_actual", fila.semestre_actual);

      if (errSel) throw errSel;

      if (existentes && existentes.length > 0) {
        const confirmar = window.confirm(
          `⚠️ Ya existe un registro para el semestre ${fila.semestre_actual}. ¿Deseas sobreescribirlo?`
        );
        if (!confirmar) {
          setCargando(false);
          return;
        }
        await supabase
          .from("predicciones_estudiantes")
          .delete()
          .eq("codigo_estudiante", usuario.codigo)
          .eq("semestre_actual", fila.semestre_actual);
      }

      // 2) Insertar nueva fila
      const { error: errIns } = await supabase.from("predicciones_estudiantes").insert([fila]);
      if (errIns) throw errIns;

      await sleep(400);

      // 3) Recuperar semestre proyectado
      const { data: filaRec, error: errRec } = await supabase
        .from("predicciones_estudiantes")
        .select("semestre_proyectado")
        .eq("codigo_estudiante", usuario.codigo)
        .eq("semestre_actual", fila.semestre_actual)
        .single();
      if (errRec) throw errRec;
      setCicloObjetivo(filaRec?.semestre_proyectado || fila.semestre_actual + 1);

      // 4) Llamar a ambas APIs (y que guarden en la fila con save=true)
      const [resR, resC] = await Promise.all([
        fetch(`${API_BASE}/predict/regresion/${usuario.codigo}?save=true`),
        fetch(`${API_BASE}/predict/continuidad/${usuario.codigo}?save=true`),
      ]);

      const jsonR = await resR.json();
      const jsonC = await resC.json();

      if (!resR.ok) throw new Error(jsonR.detail || "Error regresión");
      if (!resC.ok) throw new Error(jsonC.detail || "Error continuidad");

      const promedio = jsonR.promedio_predicho;
      const prob = jsonC.prob_riesgo;
      const riesgo = jsonC.riesgo;

      setResultadoReg(promedio);
      setResultadoCls({ prob, riesgo });

      // (Opcional) Fallback: si no usas save=true, actualiza aquí:
      // await supabase.from("predicciones_estudiantes").update({
      //   promedio_predicho: promedio,
      //   prob_riesgo_no_continuar: prob,
      //   riesgo_no_continuar: riesgo
      // }).eq("codigo_estudiante", usuario.codigo)
      //   .eq("semestre_actual", fila.semestre_actual);

    } catch (err) {
      console.error(err);
      alert("❌ Error durante el registro o la predicción.");
    } finally {
      setCargando(false);
    }
  };

  return (
    <div className={styles.wrapper}>
      <h2 className={styles.subtitulo}>🧠 Cuestionario de Predicción</h2>

      <form className={styles.formulario} onSubmit={handleSubmit}>
        <div className={styles.gridInputs}>
          {camposNumero.map(([name, label, extra]) => (
            <div key={name} className={styles.inputCard}>
              <label htmlFor={name}>{label}</label>
              <input
                id={name}
                type="number"
                name={name}
                value={formData[name] ?? ""}
                onChange={onChange}
                required
                {...extra}
                onWheel={(e) => e.currentTarget.blur()}
              />
            </div>
          ))}

          {camposSiNo.map(([name, label, options]) => (
            <div key={name} className={styles.inputCard}>
              <label htmlFor={name}>{label}</label>
              <select
                id={name}
                name={name}
                value={formData[name] ?? ""}
                onChange={onChangeSiNo}
                required
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
        // Si tu modal puede mostrar continuidad:
        continuidad={resultadoCls} // {prob, riesgo}
      />
    </div>
  );
};

export default PrediccionForm;
