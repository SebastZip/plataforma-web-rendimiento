import { useState } from 'react';
import styles from './PrediccionForm.module.css';
import { supabase } from '../supabaseClient';
import PrediccionModal from './PrediccionModal';

const camposTexto = [
  ['previous_sgpa', '📊 SGPA previo'],
  ['credit_completed', '✅ Créditos completados'],
  ['current_semester', '📘 Semestre actual'],
  ['monthly_income', '💰 Ingreso familiar mensual (S/.)'],
  ['average_attendance', '📊 Asistencia promedio (%)'],
  ['social_media_time', '📱 Horas diarias en redes'],
  ['hsc_year', '🎓 Año de egreso de secundaria'],
  ['skill_time', '💻 Horas diarias en habilidades'],
  ['age', '📅 Edad'],
  ['study_time', '📘 Horas de estudio diarias']
];

const camposSelect = [
  ['ever_probation', '⚠️ ¿Estuviste en probation?', ['Sí', 'No']],
  ['scholarship', '🏅 ¿Tienes beca?', ['Sí', 'No']]
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const PrediccionForm = ({ usuario }) => {
  const [formData, setFormData] = useState({ codigo_estudiante: usuario.codigo });
  const [resultado, setResultado] = useState(null);
  const [cicloObjetivo, setCicloObjetivo] = useState(null);
  const [mostrarModal, setMostrarModal] = useState(false);
  const [cargando, setCargando] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const transformarDatosParaEnvio = () => {
    const mapeo = { 'Sí': 1, 'No': 0 };
    const datos = { ...formData };
    Object.keys(datos).forEach(k => {
      if (mapeo.hasOwnProperty(datos[k])) datos[k] = mapeo[datos[k]];
    });
    return datos;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setCargando(true);
    setResultado(null);
    setMostrarModal(true);

    const datosFinales = transformarDatosParaEnvio();

    try {
      // 1. Verificar si ya existe
      const yaExiste = await supabase
        .from('predicciones_estudiantes')
        .select('id')
        .eq('codigo_estudiante', usuario.codigo)
        .eq('current_semester', datosFinales.current_semester);

      if (yaExiste.data.length > 0) {
        const confirmar = window.confirm(
          `⚠️ Ya existe una predicción para el semestre ${datosFinales.current_semester}. ¿Deseas sobreescribirla?`
        );
        if (!confirmar) {
          setCargando(false);
          return;
        }

        await supabase
          .from('predicciones_estudiantes')
          .delete()
          .eq('codigo_estudiante', usuario.codigo)
          .eq('current_semester', datosFinales.current_semester);
      }

      // 2. Insertar
      await supabase.from('predicciones_estudiantes').insert([datosFinales]);

      // 3. Esperar para asegurar persistencia
      await sleep(700);

      const { data: filaReciente } = await supabase
        .from('predicciones_estudiantes')
        .select('semestre_a_predecir')
        .eq('codigo_estudiante', usuario.codigo)
        .eq('current_semester', datosFinales.current_semester)
        .single();

      setCicloObjetivo(filaReciente?.semestre_a_predecir);

      // 4. Llamar API
      const res = await fetch(`https://plataforma-web-rendimiento.onrender.com/predecir/${usuario.codigo}`);
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail);

      const cgpa = json.cgpa_predicho;
      setResultado(cgpa);
      setCicloObjetivo(datosFinales.current_semester + 1);
      setMostrarModal(true);

      // 5. Actualizar
      await supabase
        .from('predicciones_estudiantes')
        .update({ ponderado_predecido: cgpa })
        .eq('codigo_estudiante', usuario.codigo)
        .eq('current_semester', datosFinales.current_semester);
    } catch (err) {
      alert('❌ Error durante la predicción o guardado.');
      console.error(err);
    } finally {
      setCargando(false);
    }
  };

  return (
    <div className={styles.wrapper}>
      <h2 className={styles.subtitulo}>🧠 Cuestionario de Predicción</h2>
      <form className={styles.formulario} onSubmit={handleSubmit}>
        <div className={styles.gridInputs}>
          {camposTexto.map(([name, label]) => (
            <div key={name} className={styles.inputCard}>
              <label>{label}</label>
              <input
                type="number"
                name={name}
                value={formData[name] || ''}
                onChange={handleChange}
                required
                min="0"
                onWheel={(e) => e.target.blur()}
              />
            </div>
          ))}
          {camposSelect.map(([name, label, options]) => (
            <div key={name} className={styles.inputCard}>
              <label>{label}</label>
              <select name={name} value={formData[name] || ''} onChange={handleChange} required>
                <option hidden value="">Selecciona una opción</option>
                {options.map(opt => <option key={opt}>{opt}</option>)}
              </select>
            </div>
          ))}
        </div>
        <button className={styles.botonVerde} type="submit">✅ Finalizar cuestionario</button>
      </form>

      <PrediccionModal
        mostrar={mostrarModal}
        onClose={() => setMostrarModal(false)}
        cgpa={resultado}
        ciclo={cicloObjetivo}
        formData={formData}
      />
    </div>
  );
};

export default PrediccionForm;
