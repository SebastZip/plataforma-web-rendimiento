import { useState } from 'react';
import Cabecera from './Cabecera';
import styles from './PrediccionForm.module.css';
import decorativa from '../assets/login/fondo_principal.jpeg';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_ANON_KEY
);

const enviarPrediccionSupabase = async (datos) => {
  const { data, error } = await supabase.from('predicciones_estudiantes').insert([datos]);
  if (error) throw error;
  return data;
};

const camposTexto = [
  ['codigo_estudiante', '📘 Código del estudiante'],
  ['Age', '🎂 Edad'],
  ['What was your previous SGPA?', '📊 SGPA previo'],
  ['H.S.C passing year', '📅 Año de egreso de secundaria'],
  ['Current Semester', '📚 Semestre actual'],
  ['How many hour do you study daily?', '🕓 Horas de estudio diario'],
  ['How many hour do you spent daily in social media?', '📱 Horas en redes sociales'],
  ['Average attendance on class', '🧑‍🏫 Asistencia promedio (0% - 100%)'],
  ['How many times do you seat for study in a day?', '📖 Sesiones de estudio por día'],
  ['How many hour do you spent daily on your skill development?', '🛠️ Horas en desarrollo de habilidades'],
  ['How many Credit did you have completed?', '✅ Créditos completados'],
  ['What is your monthly family income?', '💰 Ingreso familiar mensual']
];

const camposSelect = [
  ['Gender', 'Sexo', ['Masculino', 'Femenino']],
  ['Do you have meritorious scholarship ?', '¿Tiene beca?', ['Sí', 'No']],
  ['Do you use University transportation?', '¿Usa transporte universitario?', ['Sí', 'No']],
  ['What is your preferable learning mode?', 'Modo de aprendizaje preferido', ['Online', 'Offline']],
  ['Do you use smart phone?', '¿Tiene smartphone?', ['Sí', 'No']],
  ['Do you have personal Computer?', '¿Tiene computadora?', ['Sí', 'No']],
  ['Did you ever fall in probation?', '¿Ha estado en periodo de prueba académica (probation)?', ['Sí', 'No']],
  ['Did you ever got suspension?', '¿Tuvo suspensión?', ['Sí', 'No']],
  ['Do you attend in teacher consultancy for any kind of academical', '¿Consulta académica con docentes?', ['Sí', 'No']],
  ['Are you engaged with any co-curriculum activities?', '¿Actividades extracurriculares?', ['Sí', 'No']],
  ['With whom you are living with?', '¿Con quién vive?', ['Solo', 'Familia']],
  ['Do you have any health issues?', '¿Problemas de salud?', ['Sí', 'No']],
  ['Do you have any physical disabilities?', '¿Discapacidad física?', ['Sí', 'No']]
];

const PrediccionForm = ({ usuario }) => {
  const [paginaActual, setPaginaActual] = useState(0);
  const [formData, setFormData] = useState({
    codigo_estudiante: usuario.codigo,
    "Status of your English language proficiency_Basic": false,
    "Status of your English language proficiency_Intermediate": false,
    "What is your relationship status?_Married": false,
    "What is your relationship status?_Relationship": false,
    "What is your relationship status?_Single": false
  });

  const [cargando, setCargando] = useState(false);
  const [resultado, setResultado] = useState(null);

  const camposPaginados = [];
  const todosCampos = [...camposTexto, ...camposSelect];
  for (let i = 0; i < todosCampos.length; i += 5) {
    camposPaginados.push(todosCampos.slice(i, i + 5));
  }
  camposPaginados.push([]);

  const camposActuales = camposPaginados[paginaActual];

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleStatusEnglish = (value) => {
    setFormData({
      ...formData,
      "Status of your English language proficiency_Basic": value === 'Básico',
      "Status of your English language proficiency_Intermediate": value === 'Intermedio'
    });
  };

  const handleRelationshipStatus = (value) => {
    setFormData({
      ...formData,
      "What is your relationship status?_Single": value === 'Soltero/a',
      "What is your relationship status?_Relationship": value === 'En relación',
      "What is your relationship status?_Married": value === 'Casado/a'
    });
  };

  const transformarDatosParaEnvio = () => {
    const mapeoBinario = {
      'Sí': 1,
      'No': 0,
      'Masculino': 1,
      'Femenino': 0,
      'Online': 1,
      'Offline': 0,
      'Solo': 1,
      'Familia': 0
    };

    const data = { ...formData };
    for (let campo in data) {
      if (typeof data[campo] === 'string' && mapeoBinario.hasOwnProperty(data[campo])) {
        data[campo] = mapeoBinario[data[campo]];
      }
    }

    return data;
  };

  const siguientePagina = () => {
    if (paginaActual < camposPaginados.length - 1) {
      setPaginaActual(paginaActual + 1);
    }
  };

  const anteriorPagina = () => {
    if (paginaActual > 0) {
      setPaginaActual(paginaActual - 1);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const datosFinales = transformarDatosParaEnvio();
    setCargando(true);
    setResultado(null);

    try {
      await enviarPrediccionSupabase(datosFinales);

      const response = await fetch(`http://localhost:8000/predecir/${usuario.codigo}`);
      const data = await response.json();

      if (!response.ok) throw new Error(data.detail || 'Error en predicción');

      setResultado(data.resultado);

      await supabase
        .from('predicciones_estudiantes')
        .update({ performance: data.prediccion })
        .eq('codigo_estudiante', usuario.codigo);
    } catch (error) {
      console.error('❌ Error durante el proceso:', error);
      alert('Error durante la predicción.');
    } finally {
      setCargando(false);
    }
  };

  return (
    <div className={styles.formWrapper}>
      <Cabecera nombre={usuario.nombres} onLogout={() => window.location.reload()} />

      <form onSubmit={handleSubmit} className={styles.formContent}>
        <div className={styles.leftPane}>
          <h2 className={styles.sectionTitle}>🔹 A continuación, completa los siguientes datos:</h2>

          {camposActuales.map(([name, label, options]) => (
            <div key={name} className={styles.inputGroup}>
              <label>{label}</label>
              {options ? (
                <select
                  className={styles.inputFull}
                  name={name}
                  value={formData[name] || ''}
                  onChange={handleChange}
                  required
                >
                  <option hidden value="">Seleccione</option>
                  {options.map(opt => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              ) : (
                <input
                  className={styles.inputFull}
                  type="text"
                  name={name}
                  value={formData[name] || ''}
                  onChange={handleChange}
                  required
                  disabled={name === 'codigo'}
                  style={name === 'codigo' ? {
                    backgroundColor: '#f0f0f0',
                    cursor: 'not-allowed',
                    border: '1px solid #ccc',
                    color: '#666'
                  } : {}}
                />
              )}
            </div>
          ))}

          {paginaActual === camposPaginados.length - 1 && (
            <>
              <div className={styles.inputGroup}>
                <label>📘 Nivel de inglés</label>
                <select className={styles.inputFull} onChange={(e) => handleStatusEnglish(e.target.value)} required>
                  <option hidden value="">Seleccione</option>
                  <option value="Avanzado">Avanzado</option>
                  <option value="Intermedio">Intermedio</option>
                  <option value="Básico">Básico</option>
                </select>
              </div>

              <div className={styles.inputGroup}>
                <label>💍 Estado civil</label>
                <select className={styles.inputFull} onChange={(e) => handleRelationshipStatus(e.target.value)} required>
                  <option hidden value="">Seleccione</option>
                  <option value="Soltero/a">Soltero/a</option>
                  <option value="En relación">En relación</option>
                  <option value="Casado/a">Casado/a</option>
                  <option value="Comprometido/a">Comprometido/a</option>
                </select>
              </div>
            </>
          )}

          <div className={styles.botones}>
            {paginaActual > 0 && (
              <button type="button" onClick={anteriorPagina}>⬅️ Anterior</button>
            )}
            {paginaActual < camposPaginados.length - 1 ? (
              <button type="button" onClick={siguientePagina}>➡️ Siguiente</button>
            ) : (
              <button type="submit">✅ Finalizar</button>
            )}
          </div>

          {cargando && <p className={styles.loadingText}>🔄 Cargando predicción...</p>}

          {resultado && (
            <div className={styles.resultadoFinal}>
              <h3>🎉 Resultado de Predicción:</h3>
              <p className={styles.etiquetaResultado}>{resultado}</p>
            </div>
          )}
        </div>

        <div className={styles.sideImage}></div>
      </form>
    </div>
  );
};

export default PrediccionForm;
