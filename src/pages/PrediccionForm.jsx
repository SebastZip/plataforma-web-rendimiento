import { useState } from 'react';
import './LoginPage.css';

export const FormularioPrediccion = () => {
  const [formData, setFormData] = useState({
    "codigo": '',
    "Age": '',
    "What was your previous SGPA?": '',
    "H.S.C passing year": '',
    "Current Semester": '',
    "How many hour do you study daily?": '',
    "How many hour do you spent daily in social media?": '',
    "Average attendance on class": '',
    "How many times do you seat for study in a day?": '',
    "How many hour do you spent daily on your skill development?": '',
    "How many Credit did you have completed?": '',
    "What is your monthly family income?": '',
    "Gender": '',
    "Do you have meritorious scholarship ?": '',
    "Do you use University transportation?": '',
    "What is your preferable learning mode?": '',
    "Do you use smart phone?": '',
    "Do you have personal Computer?": '',
    "Did you ever fall in probation?": '',
    "Did you ever got suspension?": '',
    "Do you attend in teacher consultancy for any kind of academical problems?": '',
    "Are you engaged with any co-curriculum activities?": '',
    "With whom you are living with?": '',
    "Do you have any health issues?": '',
    "Do you have any physical disabilities?": '',
    "Status of your English language proficiency_Basic": false,
    "Status of your English language proficiency_Intermediate": false,
    "What is your relationship status?_Married": false,
    "What is your relationship status?_Relationship": false,
    "What is your relationship status?_Single": false
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === 'checkbox' ? checked : value
    });
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

  const handleSubmit = (e) => {
    e.preventDefault();
    const datosFinales = transformarDatosParaEnvio();
    console.log('📦 Datos transformados para envío:', datosFinales);
  };

  const camposTexto = [
    ['codigo', '📘 Código del estudiante'],
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
    ['Did you ever fall in probation?', '¿Tuvo probation?', ['Sí', 'No']],
    ['Did you ever got suspension?', '¿Tuvo suspensión?', ['Sí', 'No']],
    ['Do you attend in teacher consultancy for any kind of academical problems?', '¿Consulta académica con docentes?', ['Sí', 'No']],
    ['Are you engaged with any co-curriculum activities?', '¿Actividades extracurriculares?', ['Sí', 'No']],
    ['With whom you are living with?', '¿Con quién vive?', ['Solo', 'Familia']],
    ['Do you have any health issues?', '¿Problemas de salud?', ['Sí', 'No']],
    ['Do you have any physical disabilities?', '¿Discapacidad física?', ['Sí', 'No']]
  ];

  return (
    <div className="fondo-verde">
      <h2 className="form-title">📚 Plataforma de Predicción de Rendimiento Académico 📚 </h2>
      <p className="readme">
        Este formulario tiene como objetivo predecir el rendimiento académico de los estudiantes
        mediante el análisis de variables personales, académicas y sociales. Tus respuestas serán tratadas
        de forma anónima y utilizadas exclusivamente con fines de investigación educativa.
      </p>
      <form onSubmit={handleSubmit} className="form-grid">
        {camposTexto.map(([name, label]) => (
          <div key={name} className="form-group">
            <label>{label}</label>
            <input name={name} value={formData[name]} onChange={handleChange} required />
          </div>
        ))}

        {camposSelect.map(([name, label, options]) => (
          <div key={name} className="form-group">
            <label>{label}</label>
            <select name={name} value={formData[name]} onChange={handleChange} required>
              <option value="" disabled>Seleccione</option>
              {options.map(opt => (
                <option key={opt} value={opt}>{opt}</option>
              ))}
            </select>
          </div>
        ))}

        <div className="form-group">
          <label>📘 Nivel de inglés</label>
          <select onChange={(e) => handleStatusEnglish(e.target.value)} required>
            <option value="" disabled>Seleccione</option>
            <option value="Avanzado">Avanzado</option>
            <option value="Intermedio">Intermedio</option>
            <option value="Básico">Básico</option>
          </select>
        </div>

        <div className="form-group">
          <label>💍 Estado civil</label>
          <select onChange={(e) => handleRelationshipStatus(e.target.value)} required>
            <option value="" disabled>Seleccione</option>
            <option value="Soltero/a">Soltero/a</option>
            <option value="En relación">En relación</option>
            <option value="Casado/a">Casado/a</option>
          </select>
        </div>

        <div className="form-button-wrapper">
          <button type="submit" className="submit-button">📈 Predecir</button>
        </div>
      </form>
    </div>
  );
};
